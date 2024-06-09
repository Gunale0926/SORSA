import torch


def save_checkpoint(state, filename="checkpoint", metadata="metadata.pt"):
    state["model"].eval()
    state["model"].save_pretrained(f"{filename}")
    train_state = {}
    # Add other items in state
    train_state["optimizer"] = state["optimizer"].state_dict()
    train_state["scheduler"] = state["scheduler"].state_dict()
    train_state["epoch"] = state["epoch"]
    train_state["step"] = state.get("step", 0)
    print(f"Metadata saved to {metadata}")
    state["model"].train()
