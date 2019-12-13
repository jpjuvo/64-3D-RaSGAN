![RaSGan results - cherry-picked](/img/RaSGan-result.png)


## Jump to
- [**Methods**](#methods)
- [**Results**](#results)
- [**Tutorial**](#tutorial)

# Relativistic 3D Generative Adversarial Network: 64-3D-RaSGAN

Jolicoeur-Martineau [2018] showed that by introducing relativistic discriminator to standard generative adversarial network (SGAN), training is more stable and produces higher quality samples in image generation than with non-relativistic SGAN.

In the SGAN, discriminator and generator play a two-player game where the discriminator is trying to minimize the probability that the generated data is classified as real and the generator is trying to generate data that is falsely classified as real.

In relativistic SGAN (RSGAN), the discriminator is trying to minimize the probability that the generated data is classified as real more than the real data is classified as real and the generator is doing the opposite.

By altering the loss functions to the relativistic approach, GAN training is more stable and it should produce better quality samples without additional computational cost.
 
SGAN (non-saturating) loss functions [Goodfellow et al., 2014]:

![SGAN loss equations](/img/SGAN-loss-equations.PNG)

RSGAN loss functions [Jolicoeur-Martineau, 2018]:

![RSGAN loss equations](/img/RSGAN-loss-equations.PNG)

And to make the relativistic discriminator act more globally, we compute the average of the components instead of comparing to single random samples. This is called relativistic average SGAN (RaSGAN).

RaSGAN loss functions [Jolicoeur-Martineau, 2018]:

![RaSGAN loss equations](/img/RaSGAN-loss-equations.PNG)

The motivation to this work is to test if relativistic approach in GANs give significant improvements in stability and sample quality when generating 3D objects. The relativistic approach is used in 3D object generation method known as 3DGAN [Wu et al., 2016] to see whether it brings stability as complex joint data distributions over 3D objects are hard to train [E. Smith and D. Meger, 2017].

# Methods

### Data

Single object category of chairs (*train* part of the splitted data) from ModelNet10 [Wu et al., 2015].

Single object category of airplane (*train* part of the splitted data) from manually aligned ModelNet40 [Sedaghat et al., 2017][Sedaghat and Brox, 2015].

### Network architecture

![Network](/img/RaSGan_network.png)

The generator maps a 200-dimensional probabilistic and normally distributed latent vector *z* to ```64x64x64``` tensor that represents a 3D object in voxel space.
The discriminator is the mirror of the generator except it outputs a confidence value of the input being a fake.

### Optimizer

ADAM [Kingma and Ba, 2014] for both generator and discriminator with learning rates of 0.002 and 0.00005. Beta is 0.5.

### Training

Generating 3D objects is harder than discriminating if they are real or generated so the discriminator learns much faster. Decreasing the learning rate of the discriminator helps but I found it also necessary to restric discriminator's learning if it went too far ahead of the generator.
Here, discriminator is trained at every step when the loss of the generator is less than 200% of the discriminator's loss. As the minibatch size is relatively small (24), the losses are smoothed before deciding whether to train discriminator. Smoothing compensates for the higher loss variance error caused by a small minibatch.

The models were trained with RaSGAN using early stopping. Early stopping was not used for overfitting but for increasing generator's and decreasing discriminator's learning rate after the network had reached a state where the discriminator was trained only every 150 or more cycles.

#### Chair training
| Epochs | Generator's learning rate        | Discriminator's learning rate    |
| ----------- | ---------- | ---------- |
| 0 - 300| 0.002 |0.00005 |
| 300 - 1200| 0.005 | 0.00005 |
| 1200 - 1600 | 0.007 | 0.00003 |
| 1600 - 3000 | 0.008 | 0.00001 |

#### Airplane training
| Epochs | Generator's learning rate        | Discriminator's learning rate    |
| ----------- | ---------- | ---------- |
| 0 - 400| 0.0025 |0.00002 |
| 400 - 2000| 0.0075 | 0.00001 |

# Results

Compared to a non saturating standard GAN (SGAN), training RSGAN and RaSGAN was more stable. Relativistic GANs were able to train with broader range of initial learning rates for both generator and for discriminator.
However, I cannot fairly compare SGAN to it's relativistic counterparts because I did not thoroughly search for good values of SGAN's learning rates.

![Training](/img/chair-training.gif)

**Training chair**


![Training](/img/plane_training.gif)

**Training airplane**


![RaSGan chairs - random 2500-2900](/img/random-2500-2900.png)

**Randomly picked chairs from ephocs 2500 - 2900.**


![RaSGan chairs - cherry-picked](/img/RaSGan-chair.png)

**Cherry-picked chairs from epochs 2500 - 2900.**


![RaSGan airplanes - random 1300-1500](/img/random-1300-1500_plane.png)

**Randomly picked airplanes from ephocs 1300 - 1500.**


![RaSGan airplanes - cherry-picked](/img/RaSGan-plane.png)

**Cherry-picked airplanes from epochs 1300 - 1500.**

[![3D printing generated objects](https://img.youtube.com/vi/dpYt7f-oEuA/0.jpg)](https://youtu.be/dpYt7f-oEuA "3D printing generated objects")

Click the above image link to play the video.

***

# Tutorial
With these steps you can clone this repository and train your own model that produces 3D models.

### Requirements

- Python 3.6
- [TensorFlow](https://www.tensorflow.org/install/) (GPU version) version 1.7!
- [Patrick Min's Binvox software](http://www.patrickmin.com/binvox/) for converting training data to raw voxel data.
- [Blender](https://www.blender.org) for rendering generated models to images (optional).
- Python modules: wget, tqdm, matplotlib, numpy, tensorlayer==1.9

If you have pip installed, get these modules with:
```
pip3 install wget tqdm matplotlib numpy tensorlayer==1.9
```

**Note!** *GPU version of the Tensorflow is the only reasonable option for this tutorial. 3D-GANs of this size would take too long to train on any CPU.*

**Note-2!** This repository works with Tensorflow version 1.7 and Tensorlayer version 1.9. You may run into errors if you use newer versions. Thank you [chenyang1995](https://github.com/chenyang1995) for testing and reporting this!

## Steps

#### 1. Clone this project
```
cd [place_to_clone_this_project]
git clone https://github.com/jpjuvo/64-3D-RaSGAN.git
cd 64-3D-RaSGAN
```

#### 2. Download 3D model data for the training.
This python script will ask and download for you [Princeton's 10-Class orientation-aligned CAD models](http://modelnet.cs.princeton.edu/) or [Manually aligned 40-Class CAD models](https://github.com/lmb-freiburg/orion).

ModelNet10 object classes: *bathtub, bed, chair, desk, dresser, monitor, night stand, sofa, table, toilet.*
ModelNet40 object classes: *airplane, bathtub, bed, beanch, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower pot, glass box, guitar, keyboard, lamp, laptop, mantel, monitor, night stand, person, piano, plant, radio, range hood, sink, sofa, stairs, stool, table, tent, toilet, tv stand, wardrobe, vase, xbox.*

```
python download_data.py
```

#### 3. Convert CAD model data to voxel grid format.
Specify the class that you want to train eg. chair. Note that converting a single class may take a few hours.
This step requires [Patrick Min's Binvox software](http://www.patrickmin.com/binvox/)
```
python convert_data.py -m ModelNet10/chair -b binvox.exe
```
| Arguments | Help        | Example    | Required       |
| :-----------: | ---------- | ---------- | :-------------: |
| `--model_dir` **-m** | .off models directory | [base_dir]/ModelNet10/chair | Yes |
| `--binvox_binary` **-b** | Path to the Patrick Min's Binvox binary | binvox.exe | Yes |

#### 4. Train model with RaSGAN.
You may also experiment with other GAN architectures that are included in the repository.
```
python 64-3D-RaSGan.py -n chair-1 -d ModelNet10/chair/train -e 2500 -b 24 -sample 10 -save 10 -graph 10 -graph3d 10 -glr 0.0025 -dlr 0.00003
```
| Arguments | Help        | Example    | Required       |
| :---------: | ---------- | ------------ | :-------------: |
| `--name` **-n** | The name of the training run. This will be used to create folders and save models | chair-1 | Yes |
| `--data` **-d** | The location of the voxel grid training models | ModelNet10/chair/train | Yes |
| `--validation_data` **-v** | The location of the voxel grid validation models. If this is specified, discriminator's validation loss is also drawn to the graph. This can be helpful for monitoring divergence of train/test losses to prevent overfitting. | ModelNet10/chair/test | No |
| `--epochs` **-e** | Number of epochs to train | 2500 | No (default = 2500) |
| `--batchsize` **-b** | The batch size | 24 | No (default = 24) |
| -sample | How often generated obejcts are sampled and saved. | 10 | No (default = 10) |
| -save | How often the network models are saved. | 10 | No (default = 10) |
| -graph | How often the loss graphs are saved. | 10 | No (default = 10) |
| `--load` **-l** | Indicates if a previously loaded model should be loaded. | | No |
| `--load_epoch` **-le** | The epoch to number to be loaded from. |  | No |
| `--generator_learning_rate` **-glr** | The generator's learning rate. | 0.002 | No (default = 0.002) |
| `--discriminator_learning_rate` **-dlr** | The discriminators's learning rate. | 0.00005 | No (default = 0.00005) |
| -graph3d | How often the 3D graphs are saved. | 10 | No (default = 10) |

**Note!** *You may run into random CUDA errors or other memory related errors of TensorFlow if your batch size is too large for your gpu memory. If the training doesn't launch, try to decrease your bath size to 4 and increase it from there until you get to your gpu's limit. 24 is working with GTX 1080 with 8 GB of memory.*

#### 4. Generate 3D graph images (optional) 
3D scatter plot is a fast way to visualize your generation results.

**Note!** This is now included in the RaSGAN training and 3D graphs are generated automatically, but if you want to use a different colormap or lose the epoch title, use this. Just rename the existing 3D_graphs directory on your training run diectory before running this script.

![3D graph](/img/3D_graph_example.png)

```
python convert_to_graph.py -n chair-1
```
| Arguments | Help        | Example    | Required       |
| :-----------: | ---------- | ------------ | :-------------: |
| `--name` **-n** | Training run name for saving images from all  model files in that run | chair-1 | No if used with **-f** |
| `--file` **-f** | File path. Convert single *.npy model file to a 3D scatter plot graph | savepoint/chair-1/1000.npy | No if used with **-n** |
| `--colormap`**-c** | Matplotlib colormap options 0=copper, 1=viridis, 2=plasma, 3=winter, 4=cool, 5=ocean, 6=inferno, 7=coolwarm | 0 | No, default = 1 |

#### 5. Convert generated .npy models to .obj format (optional)
The .obj format is recognized by most 3D software.

```
python convert_to_obj.py -f savepoint/chair-1/1000.npy
```
| Arguments | Help        | Example    | Required       |
| :-----------: | ---------- | ------------ | :-------------: |
| `--name` **-n** | Training run name for converting all model files in that run | chair-1 | No if used with **-f** |
| `--file` **-f** | File path. Convert single *.npy model file to *.obj format | savepoint/chair-1/1000.npy | No if used with **-n** |

#### 6. Render .obj model to a .png image (optional)
This step requires [Blender](https://www.blender.org)

![3D render](/img/3D_render_example.png)

```
cd render
python render_class_view.py -m ../models/1000.obj -o test.png -b "C:/Program Files/Blender Foundation/Blender/blender.exe"
```
| Arguments | Help        | Example    | Required       |
| :-----------: | ---------- | ------------ | :-------------: |
| `--model_file` **-m** | CAD Model obj filename | ../models/1000.obj | Yes |
| `--blender_path` **-b** | Path to Blender executable | "C:/Program Files/Blender Foundation/Blender/blender.exe" | Yes |
| `--azimuth` **-a** | Azimuth angle in degrees | 50 | No (default = 50) |
| `--elevation` **-e** | Rendering camera's elevation | 30 | No (default = 30) |
| `--tilt` **-t** | Tilt angle in degrees | 0 | No (default = 0)|
| `--distance` **-d** | Rendering camera's distance from the model | 2.0 | No (default = 2.0) |
| `--output_img` **-o** | Output image filename | demo_img.png | No (default = demo_img.png) |

***

# Credits

## Code
Most of the training and visualization scripts were modified from Edward Smith's [Improved Adversarial Systems for 3D Object Generation and Reconstruction](https://github.com/EdwardSmith1884/3D-IWGAN) scripts.
I used his code as a base template for my project. I changed the network architecture, loss functions, training logic and added scripts for downloading the dataset, generating voxel files and for generating 3D graphs.

## GANs
The 64-3D-RaSGAN idea comes from combining the relativistic approach in [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734) with the 3DGAN in [Learning a Probabilistic Latent Space of Object
Shapes via 3D Generative-Adversarial Modeling](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf).

# References

A. Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard GAN. CoRR, abs/1807.00734, 2018. URL http://arxiv.org/abs/1807.00734.

I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, 2014.

Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao. 3d shapenets: A deep representation for volumetric shapes. IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015. (2015) 1912–1920

N. Sedaghat, M. Zolfaghari, E. Amiri, and T. Brox. Orientation-boosted voxel nets for 3D object recognition, British Machine Vision Conference, BMVC 2017. URL http://lmb.informatik.uni-freiburg.de/Publications/2017/SZB17a.

N. Sedaghat, and T. Brox. Unsupervised Generation of a Viewpoint Annotated Car Dataset from Videos. IEEE Conference on Computer Vision, ICCV 2015. URL http://lmb.informatik.uni-freiburg.de/Publications/2015/SB15.

J. Wu, C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in Neural Information Processing Systems, pages 82–90, 2016.

E. Smith and D. Meger. Improved Adversarial Systems for 3D Object Generation and Reconstruction. CoRR, abs/1707.09557, 2017. URL http://arxiv.org/abs/1707.09557.

D.P. Kingma and J. Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.
