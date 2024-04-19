# EC-SLAM: Real-time Dense Neural RGB-D SLAM System with Effectively Constrained Global Bundle Adjustment

We present EC-SLAM, a real-time dense RGB-D simultaneous localization and mapping (SLAM) system utilizing Neural Radiance Fields (NeRF). Recent NeRF-based SLAM systems have shown promising results but have yet to exploit NeRF's ability to constrain pose optimization fully. Our system leverages an effectively constrained global bundle adjustment (BA) strategy to utilize NeRF's implicit loop closure correction ability, strengthening the constraints of the keyframes most relevant to the optimized current frame and enhancing the tracking accuracy. Furthermore, we mitigate the impact of random sampling in NeRF through a feature-based and uniform sampling strategy that reduces the number of ineffective constraint points for pose optimization. EC-SLAM represents the map using sparse parametric encodings and truncated signed distance field (TSDF) for efficient fusion, achieving lower model parameters and faster convergence speed. Extensive evaluation on Replica, ScanNet, and TUM datasets demonstrates state-of-the-art performance, with up to 50\% higher tracking precision, 21 Hz runtime, and improved reconstruction accuracy due to accurate pose estimation. Our code will be open-sourced upon the acceptance of the paper. 

## 🔨 Running and Evaluating EC-SLAM

Here we elaborate on how to load the necessary data, configure EC-SLAM for your use-case, 
debug it, and how to reproduce the results mentioned in the paper.

  <details>
  <summary><b>Downloading the Data</b></summary>
  For downloading Replica, follow the procedure described on <a href="https://github.com/kxhit/vMAP">vmap</a>.<br>
  </details>

  <details>
  <summary><b>Running the code</b></summary>
  Start the system with the command:

  ```
  python run.py configs/<dataset_name>/<config_name>
  ```
  For example:
  ```
  python run.py configs/Replica/room0.yaml
  ```
  </details> 

  <details>
  <summary><b>Evaluating</b></summary>
    After running the code, you can see output/dataset_name/ for results.
  </details>

