"""JAX translation of the supplied PyTorch FB learner.

Implements the reference pairwise loss, target softmax policy, RBF inputs,
Cauchy-ball z sampling, zero-z training, and Polyak target averaging.
Pure JAX with an internal Adam optimizer (no Flax/Optax dependency).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp

from .features import Features, transform
from .networks import init_mlp, forward, backward


@dataclass(frozen=True)
class FBConfig:
    obs_dim: int = 2
    n_actions: int = 4
    embed_dim: int = 50
    hidden_dim: int = 256
    num_layers: int = 2  # additional hidden layers AFTER the first, as in source
    learning_rate: float = 5e-4
    discount_factor: float = 0.99
    reg_coef: float = 1.0
    temperature: float = 200.0  # reference uses logits / temperature
    polyak_coef: float = 0.95   # 95% target retention per update, as in source
    zero_z_p: float = 0.25
    z_sampling: str = "cauchy"  # "cauchy" or "replay_goal" (z = B(goal))
    terminate_on_goal: bool = False  # only valid for replay_goal z sampling
    goal_tolerance: float = 0.1  # Euclidean distance in original x,y space
    backward_input: str = "state"  # "state" (reference) or "state_action"
    batch_size: int = 128
    max_grad_norm: float | None = 10.0

    def __post_init__(self):
        if not (self.obs_dim > 0 and self.n_actions > 0 and self.embed_dim > 0
                and self.hidden_dim > 0 and self.num_layers >= 0 and self.batch_size > 0):
            raise ValueError("Network dimensions and batch_size must be valid")
        if not (self.learning_rate > 0 and 0 <= self.discount_factor <= 1
                and self.temperature > 0 and 0 <= self.polyak_coef <= 1
                and 0 <= self.zero_z_p <= 1):
            raise ValueError("Invalid FB hyperparameter")
        if self.z_sampling not in ("cauchy", "replay_goal"):
            raise ValueError("z_sampling must be cauchy or replay_goal")
        if self.backward_input not in ("state", "state_action"):
            raise ValueError("backward_input must be state or state_action")
        if self.goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be positive")
        if self.terminate_on_goal and self.z_sampling != "replay_goal":
            raise ValueError("terminate_on_goal requires z_sampling: replay_goal")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or null")


def init_adam(params):
    zeros = jax.tree.map(jnp.zeros_like,params)
    return {"m":zeros,"v":zeros,"t":jnp.array(0,dtype=jnp.int32)}


def _clip_by_global_norm(grads, max_norm):
    leaves = jax.tree.leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(g*g) for g in leaves))
    if max_norm is not None:
        factor = jnp.minimum(1.0,max_norm/(norm+1e-8))
        grads = jax.tree.map(lambda g:g*factor,grads)
    return grads,norm


def adam_update(params, grads, state, lr):
    t = state["t"] + 1
    m = jax.tree.map(lambda m,g:0.9*m+0.1*g,state["m"],grads)
    v = jax.tree.map(lambda v,g:0.999*v+0.001*(g*g),state["v"],grads)
    mh = jax.tree.map(lambda x:x/(1-0.9**t),m)
    vh = jax.tree.map(lambda x:x/(1-0.999**t),v)
    p = jax.tree.map(lambda w,a,b:w-lr*a/(jnp.sqrt(b)+1e-8),params,mh,vh)
    return p,{"m":m,"v":v,"t":t}


def sample_cauchy_ball(key, batch_size, embed_dim, zero_probability=0):
    """Sample z = sqrt(d) * random unit direction * Cauchy(0,0.5)."""
    k1,k2,k3 = jax.random.split(key,3)
    directions = jax.random.normal(k1,(batch_size,embed_dim))
    directions = directions / (jnp.linalg.norm(directions,axis=-1,keepdims=True)+1e-10)
    u = jax.random.uniform(k2,(batch_size,1),minval=1e-6,maxval=1-1e-6)
    scales = 0.5*jnp.tan(jnp.pi*(u-0.5))
    z = jnp.sqrt(float(embed_dim))*directions*scales
    zeros = jax.random.bernoulli(k3,zero_probability,(batch_size,))
    return jnp.where(zeros[:,None],jnp.zeros_like(z),z),zeros


def _b_embedding(params, features_obs, config, actions=None):
    """Compute B(s) or B(s,a); when a is omitted for action-conditioned B,
    marginalise uniformly across actions to represent a state-only goal.
    """
    if config.backward_input == "state":
        return backward(params, features_obs)
    if actions is not None:
        one_hot = jax.nn.one_hot(actions, config.n_actions, dtype=features_obs.dtype)
        return backward(params, jnp.concatenate((features_obs, one_hot), axis=-1))
    b, d = features_obs.shape
    broadcast_features = jnp.broadcast_to(features_obs[:, None, :], (b, config.n_actions, d))
    one_hot = jnp.broadcast_to(jnp.eye(config.n_actions, dtype=features_obs.dtype)[None, :, :],
                               (b, config.n_actions, config.n_actions))
    concatenated = jnp.concatenate((broadcast_features, one_hot), axis=-1)
    return backward(params, concatenated.reshape((b*config.n_actions, -1))).reshape(
        (b, config.n_actions, config.embed_dim)).mean(axis=1)


def goal_bootstrap_mask(next_obs, goal_obs, tolerance, terminals=None):
    """1 except when the physical next state reaches this row's sampled goal."""
    distances = jnp.linalg.norm(next_obs - goal_obs, axis=-1)
    mask = (distances > tolerance).astype(jnp.float32)
    if terminals is not None:
        mask = mask * (1.0 - terminals)
    return mask


def fb_losses(params, target_params, batch, other_obs, key, config:FBConfig, features:Features,
              fixed_z=None, goal_obs=None, other_actions=None):
    obs = transform(batch["obs"],features)
    nxt = transform(batch["next_obs"],features)
    other = transform(other_obs,features)
    actions = batch["actions"]
    mask = batch["next_action_mask"]
    terminals = batch["terminals"]
    n = config.batch_size
    if fixed_z is not None:
        # Explicit z overrides either stochastic training mode (for parity tests).
        z = jnp.asarray(fixed_z, dtype=jnp.float32)
    elif config.z_sampling == "replay_goal":
        if goal_obs is None:
            raise ValueError("replay_goal mode requires a separate goal_obs minibatch")
        # goal_obs is sampled independently from replay. No gradients through
        # the goal encoder when constructing a policy-conditioning vector.
        z = jax.lax.stop_gradient(
            _b_embedding(params["backward"], transform(goal_obs, features), config))
    else:
        z, _ = sample_cauchy_ball(key,n,config.embed_dim,config.zero_z_p)
    zero = jnp.all(z == 0, axis=-1)

    # Target F and B must not backpropagate into online parameters.
    next_f = forward(target_params["forward"],nxt,z,config.embed_dim,config.n_actions)
    q_next = jnp.einsum("bad,bd->ba",next_f,z)
    q_next = jnp.where(zero[:,None],0.0,q_next)
    logits = jnp.where(mask,q_next/config.temperature,-1e9)
    pi = jax.nn.softmax(logits,axis=-1)
    next_f_pi = jnp.einsum("bad,ba->bd",next_f,pi)
    if config.backward_input == "state_action" and other_actions is None:
        raise ValueError("state_action backward mode requires independent other_actions")
    other_b_target = _b_embedding(target_params["backward"],other,config,other_actions)
    target_matrix = jax.lax.stop_gradient(next_f_pi @ other_b_target.T)

    all_f = forward(params["forward"],obs,z,config.embed_dim,config.n_actions)
    f = jnp.take_along_axis(all_f,actions[:,None,None],axis=1)[:,0,:]
    b = _b_embedding(params["backward"],obs,config,actions)
    b_other = _b_embedding(params["backward"],other,config,other_actions)
    diag = jnp.sum(f*b,axis=-1)
    matrix = f @ b_other.T
    # Goal-dependent termination changes only the bootstrapped future-occupancy term.
    # The diagonal/current-occupancy reward contribution remains unchanged.
    if config.terminate_on_goal:
        if goal_obs is None:
            raise ValueError("terminate_on_goal requires goal_obs, even with fixed_z")
        # Physical coordinates, not z (which has dimension embed_dim).
        bootstrap = goal_bootstrap_mask(batch["next_obs"], goal_obs,
                                        config.goal_tolerance, terminals)
    else:
        bootstrap = 1.0 - terminals
    delta = matrix - config.discount_factor*bootstrap[:,None]*target_matrix
    td_loss = 0.5*jnp.mean(delta**2)-jnp.mean(diag)

    # This intentionally reproduces the *asymmetric detached* regularizer in
    # the provided forward_backward.py, rather than silently changing its loss.
    cross = b @ b_other.T
    cross_detached_other = b @ jax.lax.stop_gradient(b_other).T
    diagonal_detached = jnp.sum(b*jax.lax.stop_gradient(b),axis=-1)
    reg_loss = jnp.mean(cross_detached_other*jax.lax.stop_gradient(cross)) - jnp.mean(diagonal_detached)
    total_loss = td_loss + config.reg_coef*reg_loss
    return total_loss,{"loss":total_loss,"fb_loss":td_loss,"reg_loss":reg_loss,
                       "td_mse":jnp.mean(delta**2),"mean_q":jnp.mean(q_next),
                       "goal_hit_rate":jnp.mean((jnp.linalg.norm(batch["next_obs"]-goal_obs,axis=-1) <= config.goal_tolerance).astype(jnp.float32)) if config.terminate_on_goal else jnp.array(0.0)}


def train_step(params,target_params,opt_state,batch,other_obs,key,config,features,goal_obs=None,other_actions=None):
    (loss,metrics),grads = jax.value_and_grad(fb_losses,has_aux=True)(
        params,target_params,batch,other_obs,key,config,features,goal_obs=goal_obs,other_actions=other_actions)
    # Clip each network separately, matching torch.nn.utils.clip_grad_norm_ calls.
    fg, fnorm = _clip_by_global_norm(grads["forward"],config.max_grad_norm)
    bg, bnorm = _clip_by_global_norm(grads["backward"],config.max_grad_norm)
    grads = {"forward":fg,"backward":bg}
    updated, new_opt = adam_update(params,grads,opt_state,config.learning_rate)
    metrics = {**metrics,"forward_grad_norm":fnorm,"backward_grad_norm":bnorm}
    return updated,new_opt,metrics


def scanned_train_steps(params, targets, opt_state, batches, others, key, config, features, goals=None, other_actions=None):
    """All updates of an epoch in one XLA program; aggregate metrics on device.

    Target parameters remain FIXED throughout the scan; source code updates
    targets only once at the end of each epoch.
    """
    def step(carry, data):
        p,opt,key=carry
        key,sub=jax.random.split(key)
        if config.z_sampling == "replay_goal" and config.backward_input == "state_action":
            batch,other,goal,other_action=data
        elif config.z_sampling == "replay_goal":
            batch,other,goal=data
            other_action=None
        elif config.backward_input == "state_action":
            batch,other,other_action=data
            goal=None
        else:
            batch,other=data
            goal=None
            other_action=None
        p,opt,metrics=train_step(p,targets,opt,batch,other,sub,config,features,goal,other_action)
        return (p,opt,key),metrics
    samples = (batches,others)
    if config.z_sampling == "replay_goal":
        samples += (goals,)
    if config.backward_input == "state_action":
        samples += (other_actions,)
    (params,opt_state,key),history=jax.lax.scan(step,(params,opt_state,key),samples)
    mean_metrics=jax.tree.map(lambda v:jnp.mean(v,axis=0),history)
    return params,opt_state,key,mean_metrics


def device_epoch(params, targets, opt_state, replay, rollout_state, key,
                 env_config, fb_config, features, steps_per_env,
                 updates, horizon, terminal_mask, min_buffer_size):
    """Roll out, insert replay, update FB and Polyak average in one JIT call."""
    from .jax_environment import rollout
    from .jax_replay import add_many, sample
    rollout_state,data,collision_rate=rollout(rollout_state,steps_per_env,env_config,
                                             horizon,terminal_mask)
    replay=add_many(replay,data)
    def do_updates(args):
        params,targets,opt_state,key=args
        def body(carry,_):
            params,opt,key=carry
            if fb_config.z_sampling == "replay_goal":
                key,k1,k2,kgoal,kstep=jax.random.split(key,5)
                goal_ix=jax.random.randint(kgoal,(fb_config.batch_size,),0,replay.size)
                goal_obs=replay.obs[goal_ix]
            else:
                key,k1,k2,kstep=jax.random.split(key,4)
                goal_obs=None
            ix=jax.random.randint(k1,(fb_config.batch_size,),0,replay.size)
            other_ix=jax.random.randint(k2,(fb_config.batch_size,),0,replay.size)
            batch=sample(replay,ix)
            other=replay.obs[other_ix]
            other_actions=replay.actions[other_ix] if fb_config.backward_input == "state_action" else None
            params,opt,metrics=train_step(params,targets,opt,batch,other,kstep,fb_config,
                                          features,goal_obs,other_actions)
            return (params,opt,key),metrics
        (params,opt_state,key),metrics=jax.lax.scan(body,(params,opt_state,key),
                                                    None,length=updates)
        mean=jax.tree.map(lambda v:jnp.mean(v,axis=0),metrics)
        targets=jax.tree.map(lambda target,p:fb_config.polyak_coef*target+
                              (1-fb_config.polyak_coef)*p,targets,params)
        return params,targets,opt_state,key,mean

    def skip_updates(args):
        p,t,o,k=args
        zero=jnp.array(jnp.nan,dtype=jnp.float32)
        metrics={name:zero for name in ('loss','fb_loss','reg_loss','td_mse',
                                        'mean_q','goal_hit_rate','forward_grad_norm','backward_grad_norm')}
        return p,t,o,k,metrics
    params,targets,opt_state,key,metrics=jax.lax.cond(
        replay.size>=min_buffer_size,do_updates,skip_updates,
        (params,targets,opt_state,key))
    return params,targets,opt_state,replay,rollout_state,key,metrics,collision_rate


class FBAgent:
    def __init__(self,config:FBConfig=FBConfig(),features:Features=Features(),seed=0):
        self.config=config
        self.features=features
        self.key=jax.random.PRNGKey(seed)
        self.key,kf,kb=jax.random.split(self.key,3)
        dims_f = [features.output_dim+config.embed_dim]+[config.hidden_dim]*(1+config.num_layers)+[config.embed_dim*config.n_actions]
        backward_dim = features.output_dim + (config.n_actions if config.backward_input == "state_action" else 0)
        dims_b = [backward_dim]+[config.hidden_dim]*(1+config.num_layers)+[config.embed_dim]
        self.params = {"forward":init_mlp(kf,dims_f),"backward":init_mlp(kb,dims_b)}
        self.target_params = jax.tree.map(lambda x:x.copy(),self.params)
        self.opt_state=init_adam(self.params)
        self.updates=0
        self._compiled_update=jax.jit(lambda p,tp,opt,b,other,k:
            train_step(p,tp,opt,b,other,k,self.config,self.features))
        self._compiled_update_goals=jax.jit(lambda p,tp,opt,b,other,goal,k:
            train_step(p,tp,opt,b,other,k,self.config,self.features,goal))
        self._compiled_update_sa=jax.jit(lambda p,tp,opt,b,other,other_act,k:
            train_step(p,tp,opt,b,other,k,self.config,self.features,other_actions=other_act))
        self._compiled_update_goals_sa=jax.jit(lambda p,tp,opt,b,other,goal,other_act,k:
            train_step(p,tp,opt,b,other,k,self.config,self.features,goal,other_act))
        self._compiled_action=jax.jit(lambda p,obs,z:
            forward(p,transform(obs,self.features),z,self.config.embed_dim,self.config.n_actions))
        self._compiled_backward=jax.jit(lambda p,obs,actions:_b_embedding(p,transform(obs,self.features),self.config,actions))
        self._compiled_many=jax.jit(lambda p,tp,opt,b,o,k:
            scanned_train_steps(p,tp,opt,b,o,k,self.config,self.features))
        self._compiled_many_goals=jax.jit(lambda p,tp,opt,b,o,g,k:
            scanned_train_steps(p,tp,opt,b,o,k,self.config,self.features,goals=g))
        self._compiled_many_sa=jax.jit(lambda p,tp,opt,b,o,oa,k:
            scanned_train_steps(p,tp,opt,b,o,k,self.config,self.features,other_actions=oa))
        self._compiled_many_goals_sa=jax.jit(lambda p,tp,opt,b,o,g,oa,k:
            scanned_train_steps(p,tp,opt,b,o,k,self.config,self.features,goals=g,other_actions=oa))

    def update(self,buffer):
        batch = buffer.sample(self.config.batch_size)
        other_batch = buffer.sample(self.config.batch_size)
        other_obs = other_batch["obs"]
        self.key,stepkey=jax.random.split(self.key)
        goal_obs = buffer.sample(self.config.batch_size)["obs"] if self.config.z_sampling == "replay_goal" else None
        if self.config.backward_input == "state_action":
            if goal_obs is not None:
                self.params,self.opt_state,metrics = self._compiled_update_goals_sa(
                    self.params,self.target_params,self.opt_state,batch,other_obs,goal_obs,
                    other_batch["actions"],stepkey)
            else:
                self.params,self.opt_state,metrics = self._compiled_update_sa(
                    self.params,self.target_params,self.opt_state,batch,other_obs,
                    other_batch["actions"],stepkey)
        elif goal_obs is not None:
            self.params,self.opt_state,metrics = self._compiled_update_goals(
                self.params,self.target_params,self.opt_state,batch,other_obs,goal_obs,stepkey)
        else:
            self.params,self.opt_state,metrics = self._compiled_update(
                self.params,self.target_params,self.opt_state,batch,other_obs,stepkey)
        self.updates += 1
        return {k:float(v) for k,v in metrics.items()}

    def update_many(self, buffer, updates):
        """Batch CPU replay samples then scan all updates in one JIT dispatch.

        Returns one dictionary of mean metrics; the original update() remains
        available for debugging and backwards compatibility.
        """
        if updates < 1:
            raise ValueError("updates must be positive")
        with_goals = self.config.z_sampling == "replay_goal"
        with_actions = self.config.backward_input == "state_action"
        sampled = buffer.sample_many(updates,self.config.batch_size,
                                     with_goals=with_goals,with_other_actions=with_actions)
        batches,others = sampled[:2]
        goal_idx = 2 if with_goals else None
        actions_idx = 2 + int(with_goals) if with_actions else None
        if with_goals and with_actions:
            self.params,self.opt_state,self.key,metrics=self._compiled_many_goals_sa(
                self.params,self.target_params,self.opt_state,batches,others,
                sampled[goal_idx],sampled[actions_idx],self.key)
        elif with_goals:
            self.params,self.opt_state,self.key,metrics=self._compiled_many_goals(
                self.params,self.target_params,self.opt_state,batches,others,sampled[goal_idx],self.key)
        elif with_actions:
            self.params,self.opt_state,self.key,metrics=self._compiled_many_sa(
                self.params,self.target_params,self.opt_state,batches,others,sampled[actions_idx],self.key)
        else:
            self.params,self.opt_state,self.key,metrics=self._compiled_many(
                self.params,self.target_params,self.opt_state,batches,others,self.key)
        self.updates+=updates
        return {name:float(value) for name,value in metrics.items()}

    def soft_update(self):
        keep=self.config.polyak_coef
        self.target_params=jax.tree.map(lambda target,online:keep*target+(1-keep)*online,
                                         self.target_params,self.params)

    def embed_B(self,obs,actions=None):
        """B(s) in state-only mode. With B(s,a), supply actions for occupancy
        of state-action pairs or omit to average over all actions for a state goal.
        """
        arr = np.asarray(obs,dtype=np.float32).reshape(-1,self.config.obs_dim)
        act = None if actions is None else np.asarray(actions,dtype=np.int32).reshape(-1)
        if act is not None and len(act) != len(arr):
            raise ValueError("Expected one action per observation")
        return np.asarray(self._compiled_backward(self.params["backward"],arr,act))

    def embed_F(self,obs,z=None):
        arr=np.asarray(obs,dtype=np.float32).reshape(-1,self.config.obs_dim)
        if z is None:
            z=np.zeros((len(arr),self.config.embed_dim),dtype=np.float32)
        else:
            z=np.asarray(z,dtype=np.float32).reshape(-1,self.config.embed_dim)
            if len(z)==1 and len(arr)>1:
                z=np.repeat(z,len(arr),axis=0)
        if len(z)!=len(arr):
            raise ValueError("One z per observation, or a single broadcast z")
        return np.asarray(self._compiled_action(self.params["forward"],arr,z))

    def action_values(self,obs,z):
        f=self.embed_F(obs,z)
        z=np.asarray(z,dtype=np.float32).reshape(-1,self.config.embed_dim)
        return np.einsum("bad,bd->ba",f,np.broadcast_to(z,(f.shape[0],z.shape[1])))

    def act(self,obs,z,epsilon=0.0,available_actions=None,rng=None):
        if not 0<=epsilon<=1:
            raise ValueError("epsilon must be between zero and one")
        allowed=np.arange(self.config.n_actions,dtype=np.int32) if available_actions is None else np.asarray(available_actions,dtype=np.int32)
        if len(allowed)==0:
            raise ValueError("No available actions")
        rng=rng if rng is not None else np.random.default_rng()
        if rng.random()<epsilon:
            return int(rng.choice(allowed))
        q=self.action_values(obs,z)[0]
        return int(allowed[np.argmax(q[allowed])])

    def export_sr(self,observations,available_actions=None,clip=True):
        """N×N estimated successor density under the z=0 (uniform) policy.

        Averages F over available actions like the supplied notebook. Values are
        reference-density estimates, NOT transition probabilities or distances.
        """
        obs=np.asarray(observations,dtype=np.float32).reshape(-1,self.config.obs_dim)
        f=self.embed_F(obs)
        if available_actions is None:
            mask=np.ones((len(obs),self.config.n_actions),dtype=bool)
        else:
            mask=np.asarray(available_actions,dtype=bool)
        if mask.shape != (len(obs),self.config.n_actions) or not mask.any(axis=1).all():
            raise ValueError("Invalid action mask shape or empty row")
        fmean=(f*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1,keepdims=True)
        sr=fmean@self.embed_B(obs).T
        return np.maximum(sr,0) if clip else sr

    def save(self,path):
        """Safe NPZ checkpoint of parameters, targets, optimizer, config, PRNG key."""
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays={}
        for name,tree in (("params",self.params),("targets",self.target_params),
                          ("m",self.opt_state["m"]),("v",self.opt_state["v"])):
            for net,layers in tree.items():
                for i,layer in enumerate(layers):
                    for weight in ("w","b"):
                        arrays[f"{name}.{net}.{i}.{weight}"]=np.asarray(layer[weight])
        arrays["optimizer_t"]=np.asarray(self.opt_state["t"])
        arrays["jax_key"]=np.asarray(self.key)
        arrays["metadata"]=np.array(json.dumps({"format_version":2,"config":asdict(self.config),
            "features":asdict(self.features),"updates":self.updates}))
        with path.open("wb") as handle:
            np.savez_compressed(handle,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as arrays:
            info=json.loads(str(arrays["metadata"]))
            agent=cls(FBConfig(**info["config"]),Features(**info["features"]))
            def unpack(name,template):
                return {net:tuple({weight:jnp.asarray(arrays[f"{name}.{net}.{i}.{weight}"])
                                     for weight in ("w","b")}
                                    for i in range(len(layers)))
                        for net,layers in template.items()}
            agent.params=unpack("params",agent.params)
            agent.target_params=unpack("targets",agent.target_params)
            agent.opt_state={"m":unpack("m",agent.params),"v":unpack("v",agent.params),
                             "t":jnp.asarray(arrays["optimizer_t"])}
            # Version 1 used action-major raw output weights. Reorder the final
            # layer (including Adam moments) so old checkpoints keep the same
            # F(s,a,z) function under the PyTorch-compatible version-2 layout.
            if info.get("format_version",1) == 1:
                d, a = agent.config.embed_dim, agent.config.n_actions
                perm = np.array([action*d+embed for embed in range(d)
                                 for action in range(a)],dtype=np.int32)
                def migrate(tree):
                    tree = dict(tree)
                    layers = list(tree["forward"])
                    final = dict(layers[-1])
                    final["w"] = jnp.take(final["w"],perm,axis=1)
                    final["b"] = jnp.take(final["b"],perm,axis=0)
                    layers[-1] = final
                    tree["forward"] = tuple(layers)
                    return tree
                agent.params=migrate(agent.params)
                agent.target_params=migrate(agent.target_params)
                agent.opt_state={**agent.opt_state,
                                 "m":migrate(agent.opt_state["m"]),
                                 "v":migrate(agent.opt_state["v"])}
            agent.key=jnp.asarray(arrays["jax_key"])
            agent.updates=info["updates"]
        return agent
