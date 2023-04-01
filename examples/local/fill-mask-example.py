import mii

# roberta
name = "roberta-base"

name = "bert-base-uncased"

print(f"Deploying {name}...")

mii.deploy(task='fill-mask', model=name, deployment_name=f"{name}_deployment")
