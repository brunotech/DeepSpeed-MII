import mii

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"

# roberta
name = "roberta-large-mnli"

print(f"Deploying {name}...")

mii.deploy(
    task='text-classification',
    model=name,
    deployment_name=f"{name}_deployment",
)
