# Automatic cropper tool 
This tool is used to crop annotations from the YOLOv11 format in Roboflow

The script assumes that you have the following file structure:
```bash
.
├── raw-datasets/
│   ├── test/
│   │   ├── images/
│   │   │   └── # images
│   │   └── labels/
│   │       └── # text files
│   ├── train/
│   ├── valid/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
├── cropped-datasets/
│   ├── test/
│   │   └── [class]/
│   │       └── # cropped images
│   ├── train/
│   └── valid/
├── cropper.py
├── README.md
└── .gitignore
```