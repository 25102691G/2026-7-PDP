"""One local idealized export; no strict machine-level certification is enabled."""
from experiments.node_dp_slots import release as base_release
from . import accounting
from .mechanism import Run


def export(run, *, permit_idealized=False):
    if type(run) is not Run or run._base_run._mode != 'privacy':
        raise PermissionError('only completed privacy-mode PDP runs are eligible')
    with run._receipt.lock:
        if run._receipt.run is not run or run._receipt.published:
            raise PermissionError('unrecognized or already exported PDP run')
        if run._registry.fingerprint != run._registry_fingerprint:
            raise PermissionError('public registry mismatch')
        accounting.validate_public(run._registry, run._base_run._spec)
        payload = base_release.export(run._base_run, permit_idealized=permit_idealized)
        public = payload['configuration']
        public['base_protocol'] = public['protocol']
        public['base_epsilon_target'] = public.pop('epsilon_target')
        public['base_idealized_epsilon'] = public.pop('idealized_epsilon')
        public['base_sampling_seed'] = public.pop('sampling_seed')
        public['base_batch_seed'] = public.pop('batch_seed')
        public['base_account'] = accounting.base_account(run._base_run._spec)
        public['protocol'] = 'stable-slot-node-pdp-v1-idealized'
        public['registry'] = run._registry.public()
        public['registry_fingerprint'] = run._registry_fingerprint
        public['node_accounts'] = accounting.node_accounts(run._registry, run._base_run._spec)
        public['outer'] = dict(sampling='heterogeneous-Bernoulli-once', sampling_rounds=1,
                               hidden_randomness=True, machine_pdp_certified=False)
        public['machine_pdp_certified'] = False
        run._receipt.published = True
        return payload
