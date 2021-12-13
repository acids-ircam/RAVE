import shutil
import math
from os import path


class Print:
    def __init__(self):
        self.msg = ""

    def __call__(self, msg, end="\n"):
        print(msg, end=end)
        self.msg += msg + end


p = Print()


def header(title: str, out=p):
    term_size = shutil.get_terminal_size().columns
    term_size = min(120, term_size)
    pad_l = (term_size - len(title)) // 2

    out(term_size * "=")
    out(pad_l * " " + title.upper())
    out(term_size * "=")
    out("")


def subsection(title: str):
    p("")
    size = len(title)
    p(title.upper())
    p(size * "-")
    p("")


if __name__ == "__main__":
    header("rave command line helper", out=print)

    name = ""
    while not name:
        name = input("choose a name for the training: ")
    name = name.lower().replace(" ", "_")

    data = ""
    while not data:
        data = input("path to the .wav files: ")

    preprocessed = ""
    while not preprocessed:
        preprocessed = input("temporary folder (fast drive): ")

    sampling_rate = input("sampling rate (defaults to 48000): ")
    multiband_number = input("multiband number (defaults to 16): ")
    n_signal = input("training example duration (defaults to 65536 samples): ")
    capacity = input("model capacity (defaults to 6): ")
    warmup = input("number of steps for stage 1 (defaults to 1000000): ")
    prior_resolution = input("prior resolution (defaults to 32): ")
    fidelity = input("reconstruction fidelity (defaults to 0.95): ")
    no_latency = input("latency compensation (defaults to false): ")

    header(f"{name}: training instructions")
    subsection("train rave")

    p("Train rave (both training stages are included)")
    p("")

    cmd = "python train_rave.py "
    cmd += f"--name {name} "
    cmd += f"--wav {data} "
    prep_rave = path.join(preprocessed, name, "rave")
    cmd += f"--preprocessed {prep_rave} "

    if sampling_rate:
        cmd += f"--sr {sampling_rate} "
    if multiband_number:
        cmd += f"--data-size {multiband_number} "
    if n_signal:
        cmd += f"--n-signal {n_signal} "
    if capacity:
        cmd += f"--capacity {2**int(capacity)} "
    if warmup:
        cmd += f"--warmup {warmup} "
    if no_latency:
        cmd += f"--no-latency {no_latency.lower()} "

    p(cmd)
    p("")

    p("You can follow the training using tensorboard")
    p("")
    p("tensorboard --logdir . --bind_all")
    p("")
    p("Once the training has reached a satisfactory state, kill it (ctrl + C)")

    subsection("train prior")

    p(
        f"Export the latent space trained on {name}.",
        end="\n\n",
    )

    cmd = "python export_rave.py "

    run = path.join("runs", name, "rave")
    cmd += f"--run {run} "
    cmd += f"--cached false "
    if fidelity:
        cmd += f"--fidelity {fidelity} "
    cmd += f"--name {name}"

    p(cmd)
    p("")

    p(
        f"Train the prior model.",
        end="\n\n",
    )

    cmd = "python train_prior.py "

    if prior_resolution:
        cmd += f"--resolution {prior_resolution} "

    cmd += f"--pretrained-vae rave_{name}.ts "
    prep_prior = path.join(preprocessed, name, "prior")
    cmd += f"--preprocessed {prep_prior} "
    cmd += f"--wav {data} "

    if n_signal:
        cmd += f"--n-signal {n_signal} "

    cmd += f"--name {name}"

    p(cmd)
    p("")
    p("Once the training has reached a satisfactory state, kill it (ctrl + C)")

    p("")
    header("export to max msp (coming soon)")

    p("In order to use both **rave** and the **prior** model inside max/msp, we have to export them using **cached convolutions**."
      )
    p("")

    cmd = "python export_rave.py "

    run = path.join("runs", name, "rave")
    cmd += f"--run {run} "
    cmd += f"--cached true "
    if fidelity:
        cmd += f"--fidelity {fidelity} "
    cmd += f"--name {name}_rt"
    p(cmd)
    cmd = "python export_prior.py "

    run = path.join("runs", name, "prior")
    cmd += f"--run {run} "
    cmd += f"--name {name}_rt"

    p(cmd)

    cmd = "python combine_models.py "
    cmd += f"--prior prior_{name}_rt.ts "
    cmd += f"--rave rave_{name}_rt.ts "
    cmd += f"--name {name}"

    p(cmd)

    with open(f"instruction_{name}.txt", "w") as out:
        out.write(p.msg)