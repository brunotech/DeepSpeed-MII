import mii

mii_configs = {
    "dtype": "fp16",
    "tensor_parallel": 8,
    "port_number": 50050,
    "checkpoint_dict": {
        "checkpoints": [f'bloom-mp_0{i}.pt' for i in range(8)],
        "parallelization": "tp",
        "version": 1.0,
        "type": "BLOOM",
    },
}
name = "bigscience/bloom"

mii.deploy(
    task='text-generation',
    model=name,
    deployment_name=f"{name}_deployment",
    deployment_type=mii.constants.DeploymentType.AML,
    mii_config=mii_configs,
)
