
```
{
    "info"                  : info, 
    "images"                : [image], 
    "annotations"           : [annotation],
    "categories"            : [category], 
    "super_categories"      : [super_category], 
    "affordances"           : [affordance]
}

```

```
info {
    "year"                  : int, 
    "version"               : str, 
    "description"           : str, 
    "url"                   : str
}

image {
    "id"                    : int, 
    "width"                 : int, 
    "height"                : int, 
    "file_name"             : str,
    "depth_name"            : str
}

category {
    "id"                    : int, 
    "name"                  : str,
    "supercategory"         : str
}

super_category {
    "id"                    : int, 
    "name"                  : str
}

affordance {
    "id"                    : int, 
    "name"                  : str
}

annotation {
    "id"                    : int, 
    "image_id"              : int, 
    "category_id"           : int, 
    "segmentation"          : RLE or [polygon], 
    "area"                  : float, 
    "bbox"                  : [x,y,width,height], 
    "iscrowd"               : 0 or 1,
    "affordances"           : [annot_affordance],
    "grasps"                : [cx,cy,w,h,t],
    "grasps_id"             : [int]
}
```

```
annot_affordance {
    "aff_id"                : int,
    "segmentation":         : RLE or [polygon]
}
```