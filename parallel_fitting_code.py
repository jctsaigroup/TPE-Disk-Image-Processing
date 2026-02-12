# PARALLEL FITTING CODE - Copy this into a new cell in your notebook
# Insert this BEFORE your existing fitting loop

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_frame(frame, F_bond_pred, F_traj, IMG_DIR, fsigma=225):
    """Process a single frame - this function will be called in parallel"""
    frame_results = []
    
    image_path = os.path.join(IMG_DIR, 'Ib_'+str(frame)+'.png')
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: Could not load image for frame {frame}")
        return frame_results
    
    frame_data = F_bond_pred[F_bond_pred['frame'] == frame]
    
    for particle_id, pdata in frame_data.groupby('i'):
        z = len(pdata)
        pdata = pdata.reset_index(drop=True)
        
        if z == 1:
            continue
            
        try:
            GrayImg = get_disk_img(pdata, img)
            
            if len(GrayImg.shape) == 3:
                GrayImg = cv2.cvtColor(GrayImg, cv2.COLOR_BGR2GRAY)
            
            GrayImg = GrayImg.astype(np.float32) / 255.0 if GrayImg.max() > 1 else GrayImg.astype(np.float32)
            
            pdata_out = pdata.copy()
            rm = pdata.iloc[0]['ri']/37*6/1000
            img_size = GrayImg.shape[0]
            
            # Initial guesses
            alphas = pdata['angle_pred'].to_numpy().copy()
            betas = pdata['beta'].to_numpy().copy()
            forces = pdata['force_pred'].to_numpy().copy()
            
            # Convert to GPU tensors
            f0_gpu = torch.tensor(forces.tolist(), dtype=torch.float32, device='cuda')
            alpha0_gpu = torch.tensor(alphas.tolist(), dtype=torch.float32, device='cuda')
            betas_gpu = torch.tensor(betas.tolist(), dtype=torch.float32, device='cuda')
            GrayImg_gpu = torch.tensor(GrayImg, dtype=torch.float32, device='cuda')
            
            # Fit
            res = fit_disk_residue(GrayImg_gpu, z, fsigma, rm, img_size, 
                                  f0_gpu, alpha0_gpu, betas_gpu, 
                                  tol=1e-3, patience=20, lr=2e-2, n_iter=1000)
            f_fit, alpha_fit, fitted_loss, loss_hist = res
            
            # Store results
            pdata_out['force'] = f_fit
            pdata_out['alpha'] = alpha_fit
            pdata_out['fitLoss'] = fitted_loss.cpu().numpy()
            
            frame_results.append(pdata_out)
                
        except Exception as e:
            print(f"Error processing frame {frame}, particle {particle_id}: {e}")
            continue
    
    return frame_results


# ============== PARALLEL PROCESSING CONFIGURATION ==============
fsigma = 225
num_workers = 3  # Adjust based on GPU memory (2-4 is usually good)
frames_to_process = sorted(F_bond_pred['frame'].unique())  # Process all frames
# frames_to_process = list(range(1, 11))  # Or test with first 10 frames

print(f"Processing {len(frames_to_process)} frames with {num_workers} parallel workers...")

all_frame_data = []

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Submit all frames for processing
    future_to_frame = {
        executor.submit(process_single_frame, frame, F_bond_pred, F_traj, IMG_DIR, fsigma): frame 
        for frame in frames_to_process
    }
    
    # Collect results as they complete
    completed = 0
    for future in as_completed(future_to_frame):
        frame = future_to_frame[future]
        try:
            frame_results = future.result()
            all_frame_data.extend(frame_results)
            completed += 1
            print(f"\rCompleted: {completed}/{len(frames_to_process)} frames", end='', flush=True)
        except Exception as e:
            print(f"\nError processing frame {frame}: {e}")

print(f"\n\nDone! Processed {len(all_frame_data)} particles across {len(frames_to_process)} frames")

# Combine all results
if len(all_frame_data) > 0:
    F_bond_out = pd.concat(all_frame_data, ignore_index=True)
    print(f"Total contacts with fitted forces: {len(F_bond_out)}")
else:
    print("No data was processed!")
