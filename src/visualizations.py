import os
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from PIL import Image 
from src.fairness import compute_UCPR
import colorsys 

# utils function for figure Titles and Legends
def name_to_str(name):
    if name == "pulse":
        return "PULSE"
    if name == "real":
        return "Real"
    if name == "lr":
        return "LR"
    if name == "psp":
        return "pSp"
    if name == "psp_tmp":
        return "pSp--"
    if name == "fairpsp":
        return "fair-pSp"
    if name == "ddrm":
        return "DDRM"
    if name == "posteriorSampling":
        return "Post.Samp."
    raise NotImplementedError(f"{name} not available")

def visualize_reconstructions(paths, img_names=None, num_imgs=5, resize_size=16):
    """Plot real images, downsampled images, and upsampled images. If img_names is not specified, select images randomly."""
    methods = list(paths.keys()) 
    
    if img_names == None: 
        img_names = os.listdir(paths["real"])
        img_indices = np.random.randint(len(img_names), size=num_imgs)
        img_names = [img_names[img_index] for img_index in img_indices] 
        
    fig, axes = plt.subplots(len(img_names), len(methods), figsize=(12, len(img_names) * 1.8))
    
    # set column titles
    counter = 0
    for method in methods:
        axes[0, counter].set_title(name_to_str(method))
        counter += 1

    for i, ax in enumerate(axes.ravel()): 
        ax.axis("off")  # Turn off axis labels
        methods_id = i % len(methods)
        method = methods[methods_id]     
        img_id = i // len(methods) 
        img_name = img_names[img_id] 
        if method in ["real", "lr"]:
            image_path = os.path.join(paths["real"], img_name)
            if os.path.exists(image_path):
                img = Image.open(image_path)
                if method=="lr":
                    img = img.resize((resize_size, resize_size)) 
            else:
                print(f"{image_path} not found!")
                continue 
        else: 
            image_path = os.path.join(paths[method], img_name)
            if os.path.exists(image_path):
                img = Image.open(image_path)
            else:
                img_name = img_name.replace(".jpg", ".png") # real images are stored as jpg
                image_path = os.path.join(paths[method], img_name)
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                else:
                    print(f"{image_path} not found!")
                    continue
            
        ax.imshow(img)

    plt.tight_layout()    
    
def visualize_reconstructions_comparison(paths_unfair, paths_fair, img_names=None, num_imgs=5):
    """
    Modification of visualize_reconstructions. Given the paths of UnfairFace and FairFace reconstructions, 
    it plots the reconstructions of one model (trained on UnfairFace and FairFace) next to each other.
    """
    methods = list(paths_unfair.keys()) 
    
    if img_names == None: 
        img_names = os.listdir(paths_unfair["real"])
        img_indices = np.random.randint(len(img_names), size=num_imgs)
        img_names = [img_names[img_index] for img_index in img_indices] 
    num_imgs = len(img_names)
        
    fig_height = len(img_names) 
    num_cols = len(methods) * 2 - 2 # substract 2 because LR and real occur only once
    fig, axes = plt.subplots(len(img_names), num_cols, figsize=(12, fig_height))
    subfigs = fig.subfigures(1, len(methods) - 1)
    
    for  i, subfig in enumerate(subfigs):
        if i==0:
            subfig.suptitle("Real")
            axes = subfig.subplots(num_imgs, 2)
            for j, row in enumerate(axes):
                for k, ax in enumerate(row):
                    img_id = j 
                    img_name = img_names[img_id]
                    if k == 0:
                        method = "real"
                        if j == 0:
                            ax.set_title("HR")
                    elif k == 1:
                        method = "lr"
                        if j == 0:
                            ax.set_title("LR")
                    image_path = os.path.join(paths_unfair["real"], img_name)
                    if not os.path.exists(image_path):
                        img_name = img_name.replace(".jpg", ".png")
                        image_path = os.path.join(paths_unfair["real"], img_name)
                    img = Image.open(image_path)
                    if method == "lr":
                        img = img.resize((16, 16)) 
                    ax.axis("off")
                    ax.imshow(img)
                    ax.axis("off")  # Turn off axis labels
        else:
            method = methods[i + 1]
            subfig.suptitle(name_to_str(method))
            axes = subfig.subplots(num_imgs, 2)
            for j, row in enumerate(axes):
                for k, ax in enumerate(row):
                    img_id = j 
                    img_name = img_names[img_id]
                    if (k == 0) and (j == 0):
                        ax.set_title("UFF")
                    elif (k == 1) and (j == 0) :
                        ax.set_title("FF")
                        
                    if k == 0:
                        # UFF: 
                        paths = paths_unfair
                    else:
                        # FF
                        paths = paths_fair  
                    image_path = os.path.join(paths[method], img_name)
                    if not os.path.exists(image_path):
                        img_name = img_name.replace(".jpg", ".png")
                        image_path = os.path.join(paths[method], img_name)
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                    else:
                        print(f"{image_path} not found!")
                    
                    ax.imshow(img)
                    ax.axis("off")  # Turn off axis labels
    plt.tight_layout()

def visualize_reconstructions_avg(paths, race="Black", num_imgs=10):
    """
    Plot real average images, downsampled images, and upsampled images.
    Note, the essential difference in visualize_reconstructions lies in the file names. 
    """

    methods = list(paths.keys()) 
        
    fig, axes = plt.subplots(num_imgs, len(methods), figsize=(12, num_imgs * 1.8))
    
    # set column titles
    counter = 0
    for method in methods:
        axes[0, counter].set_title(name_to_str(method))
        counter += 1

    for i, ax in enumerate(axes.ravel()): 
        methods_id = i % len(methods)
        method = methods[methods_id]     
        img_id = i // len(methods) 
        if method == "lr":
            img_name = f"{race}.jpg"
            image_path = os.path.join(paths["real"], img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(paths["real"], img_name)
            img = Image.open(image_path)
            img = img.resize((4, 4)) 
        elif method == "real":
            img_name = f"{race}.jpg"
            image_path = os.path.join(paths["real"], img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(paths["real"], img_name)
            img = Image.open(image_path)
        else: 
            img_name = f"{race}_{img_id}.jpg"
            image_path = os.path.join(paths[method], img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(paths[method], img_name)
            if os.path.exists(image_path):
                img = Image.open(image_path)
            else:
                print(f"{image_path} not found!")
            
        ax.imshow(img)
        ax.axis("off")  # Turn off axis labels

    plt.tight_layout()
    
def visualize_reconstructions_avg_comparison(paths_unfair, paths_fair, race="Black", num_imgs=10):
    methods = list(paths_unfair.keys()) 
        
    fig_height = num_imgs
    num_cols = len(methods) * 2 - 2 # -2 because LR and real occur only once
    fig, axes = plt.subplots(num_imgs, num_cols, figsize=(12, fig_height))
    subfigs = fig.subfigures(1, len(methods) - 1)
    
    for  i, subfig in enumerate(subfigs):
        if i==0:
            subfig.suptitle("Real")
            axes = subfig.subplots(num_imgs, 2)
            for j, row in enumerate(axes):
                for k, ax in enumerate(row):
                    img_id = j 
                    img_name = f"{race}.jpg"
                    if k == 0:
                        method = "real"
                        if j == 0:
                            ax.set_title("HR")
                    elif k == 1:
                        method = "lr"
                        if j == 0:
                            ax.set_title("LR")
                    image_path = os.path.join(paths_unfair["real"], img_name)
                    if not os.path.exists(image_path):
                        img_name = img_name.replace(".jpg", ".png")
                        image_path = os.path.join(paths_unfair["real"], img_name)
                    img = Image.open(image_path)
                    if method == "lr":
                        img = img.resize((4, 4)) 
                    ax.imshow(img)
                    ax.axis("off")  # Turn off axis labels
        else:
            method = methods[i + 1]
            subfig.suptitle(name_to_str(method))
            axes = subfig.subplots(num_imgs, 2)
            for j, row in enumerate(axes):
                for k, ax in enumerate(row):
                    img_id = j 
                    img_name = f"{race}_{img_id}.jpg"
                    if (k == 0) and (j == 0):
                        ax.set_title("UFF")
                    elif (k == 1) and (j == 0) :
                        ax.set_title("FF")
                        
                    if k == 0:
                        # UFF: 
                        paths = paths_unfair
                    else:
                        # FF
                        paths = paths_fair  
                    image_path = os.path.join(paths[method], img_name)
                    if not os.path.exists(image_path):
                        img_name = img_name.replace(".jpg", ".png")
                        image_path = os.path.join(paths[method], img_name)
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                    else:
                        print(f"{image_path} not found!")
                    
                    ax.imshow(img)
                    ax.axis("off")  # Turn off axis labels
    plt.tight_layout()

def visualize_reconstructions_noisy_avg(paths, path_avg, race="Black", num_imgs=10):
    methods = list(paths.keys()) 
        
    fig, axes = plt.subplots(num_imgs, len(methods), figsize=(12, num_imgs * 1.8))
    
    # set column titles
    counter = 0
    for method in methods:
        axes[0, counter].set_title(name_to_str(method))
        counter += 1

    for i, ax in enumerate(axes.ravel()): 
        methods_id = i % len(methods)
        method = methods[methods_id]     
        img_id = i // len(methods) 
        if method == "lr":
            img_name = f"{race}_{img_id}.jpg"
            image_path = os.path.join(paths["real"], img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(paths["real"], img_name)
            img = Image.open(image_path)
            img = img.resize((4, 4)) 
        elif method == "real":
            img_name = f"{race}.jpg"
            image_path = os.path.join(path_avg, img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(path_avg, img_name)
            img = Image.open(image_path)
        else: 
            img_name = f"{race}_{img_id}.jpg"
            image_path = os.path.join(paths[method], img_name)
            if not os.path.exists(image_path):
                img_name = img_name.replace(".jpg", ".png")
                image_path = os.path.join(paths[method], img_name)
            if os.path.exists(image_path):
                img = Image.open(image_path)
            else:
                print(f"{image_path} not found!")
            
        ax.imshow(img)
        ax.axis("off")  # Turn off axis labels

    plt.tight_layout()
    
def plot_performance_per_race(loss, dfs, setting, methods, ylim=None):
    # concat all loss dfs 
    loss_dfs = []
    for method in methods:
        df = dfs[setting][method]
        df["method"] = name_to_str(method) 
        loss_dfs.append(df)
    loss_df = pd.concat(loss_dfs, ignore_index=True)
    loss_df.loc[loss_df["race"]=="Latino_Hispanic", "race"] = "Latino Hispanic"
    loss_df["race_0-1"] = 1 - loss_df["race_0-1"]
    loss_df = loss_df.rename(columns={"method": "Method"})
    
    sns.set_theme(font_scale=2.5)  # Increase font size
    sns.set_style("whitegrid")
    g = sns.catplot(data=loss_df, 
                    kind="bar", 
                    x="race", 
                    y=loss, 
                    hue="Method", 
                    palette="dark", 
                    alpha=.6, 
                    height=10, 
                    aspect=2, 
                    legend=True)
    g.despine(left=True)
    g.set(xlabel=None, ylabel=None)
    
    sns.move_legend(g, "upper center", ncol=len(methods))
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    os.makedirs("plots/performance_per_race/", exist_ok=True)
    plt.savefig(f"plots/performance_per_race/loss={loss}-setting={setting}.pdf")
    
def plot_rdp(dfs, setting, methods, ylim=None):
    # concat all loss dfs 
    loss_dfs = []
    for method in methods:
        df = dfs[setting][method]
        df["method"] = name_to_str(method) 
        loss_dfs.append(df)
    loss_df = pd.concat(loss_dfs, ignore_index=True)
    loss_df.loc[loss_df["race"]=="Latino_Hispanic", "race"] = "Latino Hispanic"
    loss_df = loss_df.rename(columns={"method": "Method"})
    
    # Get correct predictions
    correct_predictions_df = loss_df.groupby(["Method", "race", "race_0-1"]).size().reset_index(name="count")
    # Reorder to make ordering of Methods consistent 
    try: 
        # Note: You may need to modify this if you consider different methods
        custom_order = ["PULSE", "pSp", "fair-pSp", "Post.Samp.", "DDRM"]
        correct_predictions_df = correct_predictions_df.set_index('Method').loc[custom_order].reset_index()
    except:
        pass
    
    correct_predictions_df = correct_predictions_df[correct_predictions_df["race_0-1"]!= 0]
    correct_predictions_df = correct_predictions_df.drop(columns=["race_0-1"])
    
    # Scale correct predictions for each method  
    method_counts = correct_predictions_df.groupby('Method')['count'].transform('sum')
    correct_predictions_df["RDP"] = correct_predictions_df["count"] / method_counts 
    
    # Ordering to keep everything consistent
    order = ["White", 
                 "Southeast Asian", 
                 "Latino Hispanic",
                 "Middle Eastern", 
                 "Black",
                 "East Asian",
                 "Indian"]
    correct_predictions_df["race"] = pd.Categorical(correct_predictions_df["race"], order)
    
    
    sns.set_theme(font_scale=2.5)  # Increase font size
    sns.set_style("whitegrid")
    g = sns.catplot(data=correct_predictions_df, 
                    kind="bar", 
                    x="race", 
                    y="RDP", 
                    hue="Method", 
                    palette="dark", 
                    alpha=.6, 
                    height=10, 
                    aspect=2, 
                    legend=True)
    g.despine(left=True)
    g.set(xlabel=None, ylabel=None)
    
    # As a reference line, we plot the uniform distribution
    uniform = 1 / loss_df['race'].nunique()
    plt.axhline(y=uniform, color='r', linestyle='--', linewidth=4) # Uniform distribution 
    
    sns.move_legend(g, "upper center", ncol=len(methods))
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    os.makedirs("plots/performance_per_race/", exist_ok=True)
    plt.savefig(f"plots/performance_per_race/rdp-setting={setting}.pdf")
    
def plot_pr(dfs, setting, methods, ylim=None):
    # concat all loss dfs 
    loss_dfs = []
    for method in methods:
        df = dfs[setting][method]
        df["method"] = name_to_str(method) 
        loss_dfs.append(df)
    
    loss_df = pd.concat(loss_dfs, ignore_index=True)
    loss_df.loc[loss_df["race_recon"]=="Latino_Hispanic", "race_recon"] = "Latino Hispanic"
    loss_df = loss_df.rename(columns={"method": "Method",
                                      "race_recon": "Race"})
    loss_df["race_0-1"] = 1 - loss_df["race_0-1"]
    
    
    sns.set_theme(font_scale=2.)  # Increase font size
    sns.set_style("whitegrid")

    plt.figure(figsize=(20,10))

    order = ["White", 
                 "Southeast Asian", 
                 "Latino Hispanic",
                 "Middle Eastern", 
                 "Black",
                 "East Asian",
                 "Indian"]
    loss_df["Race"] = pd.Categorical(loss_df["Race"], order)

    # catplot uses saturation of 0.75 instead of the "original" colors. 
    # This adjusts the color palette to make it consistent.  
    palette = sns.color_palette("dark")
    adjusted_palette = []
    for color in palette:
        h, l, s = colorsys.rgb_to_hls(*color)
        adjusted_color = colorsys.hls_to_rgb(h, l, s * 0.75)
        adjusted_palette.append(adjusted_color)
    g = sns.histplot(data=loss_df, 
                     x="Race", 
                     stat="proportion", 
                     hue="Method",
                     palette=adjusted_palette, #"dark",
                     common_norm=False,
                     multiple="dodge",
                     alpha=.6, 
                     shrink=0.7
                    )
    sns.despine(left=True)
    plt.gca().xaxis.grid(False)
    g.set(xlabel=None, ylabel=None)
    
    # As a reference line, we plot the uniform distribution
    uniform = 1 / loss_df['Race'].nunique()
    plt.axhline(y=uniform, color='r', linestyle='--', linewidth=4) # Uniform distribution 
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    os.makedirs("plots/performance_per_race/", exist_ok=True)
    plt.savefig(f"plots/performance_per_race/pr-setting={setting}.pdf")
    
def plot_ucpr(dfs, setting, methods, races, ylim=None):
    ucpr = compute_UCPR(dfs, setting, methods, races) 
    ucpr_df = pd.DataFrame(ucpr)
    ucpr_df["race"] = races
    if setting in ["fairface_avg", "unfairface_avg"]:
        ucpr_df = ucpr_df.drop(["psp", "fairpsp"], axis=1) 
    ucpr_df = ucpr_df.rename(columns={"pulse": "PULSE",
                                      "posteriorSampling": "Post.Samp.",
                                      "ddrm": "DDRM"})
    ucpr_df.loc[ucpr_df["race"]=="Latino_Hispanic", "race"] = "Latino Hispanic"


    order = ["White", 
                    "Southeast Asian", 
                    "Latino Hispanic",
                    "Middle Eastern", 
                    "Black",
                    "East Asian",
                    "Indian"]

    ucpr_df["race"] = pd.Categorical(ucpr_df["race"], order, ordered=True)
    ucpr_df = ucpr_df.sort_values(by='race')
    
    ucpr_df = pd.melt(ucpr_df, id_vars=["race"], var_name="Method")


    sns.set_theme(font_scale=4.3)  # Increase font size
    sns.set_style("whitegrid")

    plt.figure(figsize=(40,20))
    
    g = sns.barplot(data=ucpr_df, 
                    x="race", 
                    y="value", 
                    hue="Method",
                    palette="dark",
                    alpha=.6,
                    width=0.8,
                    dodge=True
                    )
   
    g.set(xlabel=None, ylabel=None)
    
    # As a reference line, we plot the uniform distribution
    uniform = 1 / ucpr_df["race"].nunique()
    plt.axhline(y=uniform, color='r', linestyle='--', linewidth=8) # Uniform distribution 
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    os.makedirs("plots/performance_per_race/", exist_ok=True)
    plt.savefig(f"plots/performance_per_race/ucpr-setting={setting}.pdf")
