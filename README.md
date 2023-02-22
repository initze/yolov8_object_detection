# yolov8_object_detection
repo for some experimental object detection tasks from aerial imagery

## Data structure
#### Scripts
`inference.py` Run inference on a full datase

#### Data
```
project
│   README.md
│   inference.py
│   
└───data
│   │───SITE_NAME
│   │   |   image1.jpg
│   │   |   image2.jpg
```

### Usage example
`inference.py --name <SITE_NAME>`
`inference.py --name 20210628-230058_17_BP_DrainedLake_1000m`
