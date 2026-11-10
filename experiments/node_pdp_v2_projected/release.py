"""Version-specific two-field whitelist; strict machine export remains blocked."""
import math
from .mechanism import Run
from . import accounting


def export(run,*,permit_idealized=False):
    if type(run) is not Run or run._mode!='privacy' or not run._completed:
        raise PermissionError('only completed v2 privacy-mode runs eligible')
    with run._receipt.lock:
        if run._receipt.run is not run or run._receipt.published:
            raise PermissionError('unrecognized/already exported v2 run')
        if not permit_idealized:
            raise PermissionError('UNRESOLVED machine proof; strict DP/PDP export blocked')
        if (len(run._parameters)!=2340 or not all(math.isfinite(v) for v in run._parameters)
            or any(v<0 for v in run._parameters[run._layout.start:run._layout.stop])):
            raise PermissionError('invalid projected fixed-dimensional parameters')
        if run._registry.fingerprint!=run._registry_fingerprint:raise PermissionError('registry mismatch')
        public=run._spec.public()
        public['base_component_protocol']=public.pop('protocol')
        public['protocol']='node_pdp_v2_projected'
        public['base_epsilon_target']=public.pop('epsilon_target')
        public['base_sampling_seed']=public.pop('sampling_seed')
        public['base_batch_seed']=public.pop('batch_seed')
        public['base_account']=accounting.base_account(run._spec)
        public['node_accounts']=accounting.node_accounts(run._registry,run._spec)
        public['registry']=run._registry.public();public['registry_fingerprint']=run._registry_fingerprint
        public['projection']=run._layout.public()
        public['outer']=dict(sampling='heterogeneous-Bernoulli-once',sampling_rounds=1,hidden_randomness=True)
        public['machine_pdp_certified']=False
        public['development_process_dp_protected']=False
        run._receipt.published=True
        return dict(parameters=list(run._parameters),configuration=public)
