# Development Environment Setup

## Docker Configuration

### Docker Image
- **Name:** `kalibr-modified`
- **Base:** ROS1 Noetic
- **Status:** Contains custom modifications for trajectory export

### Source Code Locations

**On Host Machine:**
```
~/Kalibr/src/kalibr/          # Kalibr source code (editable)
~/async_vision/               # Main project directory
├── src/                      # Custom pipeline code
├── scripts/                  # Analysis scripts
│   └── cam_trajectory.py  # Trajectory visualization
└── docs/                     # Documentation (this file)
```

**In Docker Container:**
```
/catkin_ws/src/kalibr/        # Mounted from host
/data/                        # Mounted data directory
/output/                      # Mounted output directory
```

---

## Docker Aliases

### Quick Use Alias
```bash
alias kalibr='xhost +local:root && docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v "/home/sid/NeuROAM_data:/data" kalibr-modified'
```

**Use when:** Running Kalibr with existing modifications, no code editing needed

---

### Development Alias
```bash
alias kalibr_dev='xhost +local:root && docker run -it \
  -e "DISPLAY" \
  -e "QT_X11_NO_MITSHM=1" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/home/sid/Kalibr/src/kalibr:/catkin_ws/src/kalibr:rw" \
  -v "/home/sid/NeuROAM_data:/data:rw" \
  -v "/home/sid/Kalibr:/output:rw" \
  --name kalibr_dev \
  kalibr-modified \
  /bin/bash'
```

**Use when:** Modifying Kalibr source code

**Volume mounts:**
| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 display (GUI support) |
| `~/Kalibr/src/kalibr` | `/catkin_ws/src/kalibr` | Source code (read-write) |
| `~/NeuROAM_data` | `/data` | Input datasets |
| `~/Kalibr` | `/output` | Calibration results |

---

## Development Workflow

### 1. Edit Code Locally
```bash
# Open in VS Code (on host machine, no ROS1 needed)
code ~/Kalibr/src/kalibr
```

### 2. Start Development Container
```bash
kalibr_dev
```

### 3. Rebuild After Changes
```bash
# Inside container
cd /catkin_ws
catkin build

# Or clean build if needed
catkin clean -y
catkin build
```

### 4. Test Changes
```bash
# Run calibration with your modifications
rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/your_bag.bag \
  --topics /cam0/image_raw /cam1/image_raw \
  --models pinhole-radtan pinhole-radtan \
  --target /data/aprilgrid.yaml
```

### 5. Exit and Commit
```bash
# Exit container
exit

# Commit changes (on host)
cd ~/Kalibr/src/kalibr
git status
git add modified_file.py
git commit -m "Description of changes"
```

---

## Troubleshooting

### Container Already Exists Error
```bash
# If you get "container name already in use"
docker rm kalibr_dev_container

# Then run kalibr_dev again
```

### View Running Containers
```bash
docker ps
```

### Stop Development Container
```bash
docker stop kalibr_dev_container
```

### Remove Container (Clean Slate)
```bash
docker rm kalibr_dev_container
```

### Rebuild Kalibr from Scratch
```bash
# Inside container
cd /catkin_ws
rm -rf build/ devel/
catkin build
```

---

## Git Repository Structure

### Branches
- `master`: Stable version with working modifications
- `cam-timesync`: Active development branch

### Common Git Commands
```bash
# Check status
git status

# See changes
git diff

# View commit history
git log --oneline

# Create feature branch
git checkout -b feature-name

# Switch branches
git checkout branch-name
```

---

## System Information

### Host System
- OS: Ubuntu (ROS2 installed)
- Docker version: Check with `docker --version`

### Container System
- OS: Ubuntu 20.04 (in Docker)
- ROS: ROS1 Noetic
- Python: 2.7 and 3.8

---

## Important Paths

### Data
- NeuROAM dataset: `/home/sid/NeuROAM_data/`
- Kalibr results: `/home/sid/Kalibr/raw_data/`

### Code
- Custom pipeline: `~/async_vision/src/`
- Kalibr source: `~/Kalibr/src/kalibr/`
- Trajectory scripts: `~/async_vision/scripts/`

---

## Notes
- Always edit on host machine (not inside container)
- Container is only for building and running
- Changes to mounted directories persist on host
- Git operations done on host, not in container