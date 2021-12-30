## Single Image 3D Scene Reconstruction Based on ShapeNet Models [[Project Page]](https://sjtu-cv-2021.github.io/Single-Image-3D-Reconstruction-Based-On-ShapeNet/)[[Oral Paper]](paper.pdf)

> **Single Image 3D Scene Reconstruction Based on ShapeNet Models**
Xueyang Chen*, Yifan Ren*, Yaoxu Song*
>
> *Zhiyuan College, Shanghai Jiao Tong University, Shanghai 200240, People’s Republic of China


<img src="image/image11.jpg" alt="img.jpg" width="32%" /> <img src="image/image21.png" alt="recon.png" width="32%" /> <img src="image/image8.png" alt="recon.png" width="32%" /> 
<br>
<img src="image/image23.jpg" alt="img.jpg" width="32%" /> <img src="image/image4.png" alt="recon.png" width="32%" /> <img src="image/image9.jpg" alt="recon.png" width="32%" /> 

---


### Abstract and Method
The 3D scene reconstruction task is the basis for implementing mixed reality, but traditional single-image scene reconstruction algorithms are difficult to generate regularized models. It is believed that this situation is caused by a lack of prior knowledge, so we try to introduce the model collection ShapeNet to solve this problem. Besides, our approach incorporates traditional model generation algorithms. The predicted artificial indoor objects as indicators will match models in ShapeNet. The refined models selected from ShapeNet will then replace the rough ones to produce the final 3D scene. These selected models from the model library will greatly improve the aesthetics of the reconstructed 3D scene. We test our method on the NYU-v2 dataset and achieve pleasing results.

<img src="image/image10.png" alt="method.jpg" width="100%" />

As illustrated in above figure, we propose an end-to-end method given a single image as input and the 3-D reconstruction result from the image as output. The intermediate networks we utilize were first proposed in Total3D. When given a single image with 2D bounding boxes as its input, those networks can construct a roughly 3-D bounding box and mesh for each object. These meshes are then sampled to produce point clouds of the corresponding objects. On the other side of our work, we sample a part of the model in ShapeNet to get their point clouds. Then labels of the objects being matched would serve as indicators of how to split the original point cloud set into smaller sets, each corresponding to a label. Finally, with a refined model in ShapeNet for each object, we can embed the 3-D model of each object into the 3-D scene and complete the reconstruction work.

---


### Video
<!--[![IMAGE_ALT](https://img.youtube.com/vi/3ho8UDLv-UQ/hqdefault.jpg)](https://youtube.com/watch?v=3ho8UDLv-UQ)-->
<iframe
    width="640"
    height="360"
    src="https://www.youtube.com/embed/3ho8UDLv-UQ"
    frameborder="0"
    allow="autoplay; encrypted-media"
    allowfullscreen
>
</iframe>
---


### Code and Data
We provide source codes and related data of the project on [[**Our Github Page**]](https://github.com/SJTU-CV-2021/Single-Image-3D-Reconstruction-Based-On-ShapeNet)
