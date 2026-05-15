
#
# Package imports
#
import random 
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator




#
# Random integer generation
#
random_integer = random.randint(1, 1129)
#random_integer = 832
print(random_integer)



#
# Loading LePIC Input Data
#
def load_lepic_input_file(path):
    ions = []
    neutrals = []
    reactions = []

    with open(path, "r") as f:
        lines = [line.rstrip() for line in f]
    #parse section 
    i = 0
    n = len(lines)

    def is_dash(line):
        return set(line.strip()) == {"-"}

    while i < n:
        line = lines[i].strip()

        # ---------------- IONS ----------------
        if line == "IONS":
            i += 2  # skip header line
            while not is_dash(lines[i]):
                parts = lines[i].split()
                ions.append({
                    "name": parts[0].strip("[]"),
                    "mass": float(parts[1]),
                    "charge": float(parts[2]),
                    "temp_eV": float(parts[3]),
                    "density_ratio": float(parts[4]),
                })
                i += 1
            i += 1
            continue

        # ---------------- NEUTRALS ----------------
        if line == "NEUTRALS":
            i += 2
            while not is_dash(lines[i]):
                parts = lines[i].split()
                neutrals.append({
                    "name": parts[0].strip("[]"),
                    "mass": float(parts[1]),
                    "charge": float(parts[2]),
                    "temp_eV": float(parts[3]),
                    "density_ratio": float(parts[4]),
                })
                i += 1
            i += 1
            continue

        # ---------------- REACTION ----------------
        if line == "REACTION":
            reaction = {}

            reaction["equation"] = lines[i + 1].strip()
            reaction["threshold_eV"] = float(lines[i + 2].split()[0])
            reaction["label"] = lines[i + 4].strip()

            # metadata line (keep whole thing)
            meta_idx = i + 7
            reaction["metadata"] = lines[meta_idx]

            # data starts after dashed line
            i = meta_idx + 2

            energy = []
            xs = []

            while i < n and not is_dash(lines[i]):
                parts = lines[i].split()
                if len(parts) >= 2:
                    try:
                        energy.append(float(parts[0]))
                        xs.append(float(parts[1]))
                    except ValueError:
                        pass
                i += 1

            reaction["energy"] = np.array(energy)
            reaction["cross_section"] = np.array(xs)

            reactions.append(reaction)
            i += 1
            continue

        i += 1

    return {
        "ions": ions,
        "neutrals": neutrals,
        "reactions": reactions
    }

#
# Load LePIC input data main
#
input_lepic_data = load_lepic_input_file("deuterium.dat")

#
# report LePIC load status 
#
print(len(input_lepic_data["ions"]))
print(len(input_lepic_data["neutrals"]))
print(len(input_lepic_data["reactions"]))
print("=====================================")
print("Sample reaction:")

r = input_lepic_data["reactions"][random_integer]
print(r["equation"])
print(r["energy"].min(), r["energy"].max())

print("=====================================")


#
# Load LePIC  Output Data (reactions.1 & reactions.5)
#

def load_all_lepic_output_reactions(filename, npts=360, reaction_names=None):
 
    data = np.loadtxt(filename)
    num_reactions = data.shape[0] // npts
    remainder = data.shape[0] % npts
    
    if remainder > 0:
        print(f"Warning: {remainder} lines will be ignored (not a clean multiple of {npts})")
    
    if reaction_names is None:
        reaction_names = [f"Reaction {i}" for i in range(num_reactions)]
    elif len(reaction_names) != num_reactions:
        print(f"Warning: {len(reaction_names)} names provided but {num_reactions} reactions found.")
        # Pad names to match
        if len(reaction_names) < num_reactions:
            reaction_names = reaction_names + [f"Reaction {i}" for i in range(len(reaction_names), num_reactions)]
        else:
            reaction_names = reaction_names[:num_reactions]
    
    reactions = {}
    for idx, name in enumerate(reaction_names):
        start = idx * npts
        end = start + npts
        block = data[start:end]
        E = block[:, 0]
        sigma = block[:, 1]
        reactions[name] = (E, sigma)
    
    return reactions

#
# ===== Load LePIC reactions main ===
#
if __name__ == "__main__":
    e_incident = r"C:\Users\cmack\OneDrive\Documents\s25_intern\py-projects\xsections\Deuterium\LePIC_cross_sections\reactions.1"
    H_plus_incident = r"C:\Users\cmack\OneDrive\Documents\s25_intern\py-projects\xsections\Deuterium\LePIC_cross_sections\reactions.5"

    e_reactions = load_all_lepic_output_reactions(e_incident)
    H_plus_reactions = load_all_lepic_output_reactions(H_plus_incident)

#
# report LePIC load status
#
print(f"Loaded LePIC e_reactions: {len(e_reactions)} reactions from {os.path.basename(e_incident)}")
print(f"Loaded LePIC H_plus_reactions: {len(H_plus_reactions)} reactions from {os.path.basename(H_plus_incident)}")
print("=====================================")

#
# Logic for matching reactions between datasets
#
def match_reactions(input_data, e_output_reactions, dplus_output_reactions):
    matches = []

    input_reactions = input_data.get("reactions", [])

    e_keys = list(e_output_reactions.keys())
    d_keys = list(dplus_output_reactions.keys())

    e_idx = 0
    d_idx = 0

    for i, rx in enumerate(input_reactions):
        eq = rx.get("equation", "")
        incident = None

        # --- extract incident species from input metadata --- (e.g. " [e] + [*] " ----> flagged as "e" incident)
        m = re.search(r'\[([^\]]+)\]\s*\+\s*\[', eq)
        if m:
            incident = m.group(1).strip()
        else:
            parts = re.split(r'\+|->', eq)
            if parts:
                incident = parts[0].strip().strip("[]")

        output_key = None
        output_E = None
        output_sigma = None

        if incident is not None:
            low = incident.lower() 

            #Define the two incident species electron and D+
            # -------- [e-] --------
            if low in ("e", "e-", "electron"):
                if e_idx < len(e_keys):
                    output_key = e_keys[e_idx]
                    output_E, output_sigma = e_output_reactions[output_key]
                else:
                    print(f"Warning: ran out of electron output reactions at input index {i}")
                e_idx += 1

            # -------- [D+] --------
            elif low.startswith("d+"):
                if d_idx < len(d_keys):
                    output_key = d_keys[d_idx]
                    output_E, output_sigma = dplus_output_reactions[output_key]
                else:
                    print(f"Warning: ran out of D+ output reactions at input index {i}")
                d_idx += 1

            # -------- " " --------
            else:
                print(f"Warning: unknown incident species '{incident}' at input index {i}")

        matches.append({
            "input_index": i,
            "equation": eq,
            "incident": incident,
            "output_key": output_key,
            "output_E": output_E,
            "output_sigma": output_sigma,
            "e_local_index": e_idx - 1 if incident and low in ("e", "e-", "electron") else None,
            "dplus_local_index": d_idx - 1 if incident and low.startswith("d+") else None,
        })

    # --- sanity checks ---
    if e_idx != len(e_keys):
        print(f"Warning: used {e_idx} electron outputs but file contains {len(e_keys)}")
    if d_idx != len(d_keys):
        print(f"Warning: used {d_idx} D+ outputs but file contains {len(d_keys)}")

    return matches


#
# Match input reactions to output blocks
#
mappings = match_reactions(input_lepic_data, e_reactions, H_plus_reactions)
# Print a few sample matches for verification
#for m in mappings[:10]:
#    print(f"Input #{m['input_index']}: {m['equation']} -> Output key: {m['output_key']},"
#          f" incident: {m['incident']}, output present: {m['output_E'] is not None}")


#
# Plotting a random matched reaction
#
def plot_random_match(save_path="match_plot.png"):
    
    num_inputs = len(input_lepic_data.get("reactions", []))
    if num_inputs == 0:
        print("No input reactions available to plot.")
        return

    sel_idx = random_integer % num_inputs
    rx_in = input_lepic_data["reactions"][sel_idx]
    in_E = rx_in.get("energy", np.array([]))
    in_sigma = rx_in.get("cross_section", np.array([]))

    # find mapping for this input index
    mapping = next((m for m in mappings if m["input_index"] == sel_idx), None)

    plt.figure(figsize=(8,5))
    if in_E.size and in_sigma.size:
        plt.plot(in_E, in_sigma, label=f"Input #{sel_idx} (input file)", lw=2)
    else:
        print(f"Input reaction #{sel_idx} has no data.")

    if mapping and mapping.get("output_E") is not None:
        out_E = mapping["output_E"]
        out_sigma = mapping["output_sigma"]

        # if energy grids differ, interpolate output onto input energies for direct comparison
        try:
            if in_E.size and out_E.size and not np.array_equal(in_E, out_E):
                interp = PchipInterpolator(out_E, out_sigma, extrapolate=False)
                out_on_in = interp(in_E)
                plt.plot(in_E, out_on_in, "--", label=f"Output ({mapping['output_key']}) interp", lw=1.5)
            else:
                plt.plot(out_E, out_sigma, "--", label=f"Output ({mapping['output_key']})", lw=1.5)
        except Exception as e:
            # fallback: plot raw output
            plt.plot(out_E, out_sigma, "--", label=f"Output ({mapping['output_key']})", lw=1.5)
            print("Interpolation failed:", e)
    else:
        print(f"No matched output found for input #{sel_idx} (incident: {mapping['incident'] if mapping else 'N/A'})")

    plt.xlabel("Energy (eV)")
    plt.xscale("log")
    plt.yscale("log")   
    plt.ylabel("Cross-section")
    plt.title(f"Input vs Matched Output — Input #{sel_idx}: {rx_in.get('equation','')}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()

#
# Plot a random matched reaction main
#
if __name__ == "__main__":
    plot_random_match()

