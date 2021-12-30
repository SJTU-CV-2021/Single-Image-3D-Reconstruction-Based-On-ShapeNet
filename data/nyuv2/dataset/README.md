#### Preprocess NYU-v2 data

1. Download the raw [NYU-v2 data](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts) to
```
data/nyuv2/dataset/
```
We recomand you to download the Misc part of the whold dataset. Then unzip all the files in the same directory.

2. Download the class labels of objects in SUN RGB-D images [[link](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)] to 
```
data/nyuv2/dataset/nyu_depth_v2_labeled.mat
```

3. Then, your directory tree should look like this:

```
Single-Image-3D-Reconstruction-Based-On-ShapeNet
├── data
    ├── nyuv2
		├── dataset
			├── computer_lab_0001
			├── computer_lab_0002
			├── conference_room_0001
			├── conference_room_0002
			├── dentist_office_0001
			├── dentist_office_0002
			├── dinette_0001
			├── excercise_room_0001
			├── foyer_0001
			├── foyer_0002
			├── home_storage_0001
			├── indoor_balcony_0001
			├── laundry_room_0001
			├── misc_part1.zip
			├── misc_part2.zip
			├── nyu_depth_v2_labeled.mat
			├── patio_0001
			├── printer_room_0001
			└── student_lounge_0001
```
