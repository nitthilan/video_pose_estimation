
Algorithm:
- Find beta and theta for all the frames
- Use projection to make sure the predicted poses are proper
- HMR could be used to initialize the first frames

Things to try:
- A simple filtering 
- A simple interpolation to remove noise. Fit a curve.

Need not estimate new betas since we can always use the old beta and then scale the vertices by scale factor

Find average beta and use that to initialze for all frames.
Use a batch on N frames to iterate for the pose
Make sure the 

python smplifyx/main.py --config cfg_files/fit_smplx.yaml     --data_folder /nitthilan/data/neuralbody/people_snapshot_public/female-1-casual/    --output_folder /nitthilan/data/neuralbody/people_snapshot_public/female-1-casual/smplx/female     --visualize=False     --model_folder /nitthilan/data/SMPLX/models/     --vposer_ckpt /nitthilan/data/SMPLX/V02_05/     --part_segm_fn smplx_parts_segm.pkl



Competitors:
https://www.deepmotion.com/