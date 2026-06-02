# StarDist

**StarDist** is a deep learning model for object detection and segmentation, particularly well-suited for star-convex shapes like disks. In this pipeline, StarDist2D is used to detect and segment photoelastic disks in the green fluorescence channel images.

- [StarDist GitHub](https://github.com/stardist/stardist)
- [StarDist paper](https://arxiv.org/abs/1806.03535)

## How it works

StarDist predicts a set of rays (distances from the center) for each object, allowing it to reconstruct the object boundary as a star-convex polygon. This is more robust than simple thresholding or circular Hough transforms, especially in crowded or noisy images.

## Usage in this pipeline

- Input: Green-channel images (`green_*.png`)
- We `camera_align` the green image into our standard coordinate system before feeding it to StarDist.

- The magic happens at
```python
   #PREDICT 
    mask, detail = model.predict_instances(input_image, n_tiles=model._guess_n_tiles(input_image), show_tile_progress=False)
```

The model returns a labelled `mask` object containing each pixel region detected as disk, and a unique integer label for each region. We then find the centroid of the region as the temporary disk center, and also obtain the eccentricity of each region to later filter out non-disk objects. Area is used duistinguish between large and small disks, to which we then manually assign a radius of 46 or 37 pixels, respectively.

```python
# Compute properties
    props = measure.regionprops(mask)

    # Create a list of dicts
    for region in props:
        y, x = region.centroid  # note: (row, col) = (y, x)
        records.append({
            "frame": frame,
            "x": x,
            "y": y,
            "area": region.area,
            "ecc": region.eccentricity
        })
......

df_filtered = df[(df["x"] < 1860)&(df["x"] > 100)&(df["y"] > 50)] #trim off boundary 
df_filtered = df_filtered[df_filtered["ecc"] < 0.6] #filter eccentricity #ecc filter changed to 0.6 20250923
df_filtered.loc[:,'rpx'] = 46 #add radius column
df_filtered.loc[df_filtered['area']<6000, 'rpx'] = 37
```

The detected centroids are then linked into trajectories using [Trackpy](https://soft-matter.github.io/trackpy/).