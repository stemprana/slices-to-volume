


#NOTE: File with main development that is going to be branched out for easier showcasing,
#Sections are going to identify as 'ORIGINAL' for things that are not required to be shown and 'COMMON' for shared things

#%% COMMON
# IMPORTs -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import copy
import numpy as np
import json
import napari
import matplotlib.pyplot as plt
import cv2
from skimage.transform import downscale_local_mean
from tifffile import memmap
import slice_to_volume as slc_to_vol
from pathlib import Path
%matplotlib qt5
#%%ORIGINAL
# IMPORTs -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import slice_to_volume_access as slc_to_vol_aid
from napari_animation import Animation
from napari_animation.easing import Easing

#%% COMMON 
# GET IMAGES METADATA ------------------------------------------------------------------------ 
map_path =  Path('/home/silvio/Documents/Lfiles/inputData/confocal_data/20240906/20240906_SGT264_#972_Sl.3-2/20240906_SGT264_#972_Sl.3-2_map_Cycle') # Path to low res files used for map creation
stack_path =  Path('/home/silvio/Documents/Lfiles/inputData/confocal_data/20240906/20240906_SGT264_#972_Sl.3-2/20240906_SGT264_#972_Sl.3-2_stacks_Cycle') # Path to detailed stacks 
save_path = Path('/home/silvio/Documents/Lfiles/outputData/confocal_reconstruction')

#%% ORIGINAl
# Read positioning information file  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
stack_info_0 = slc_to_vol_aid.get_stitch_info(os.path.join(stack_path,'matl.omp2info'))
map_info_0 = slc_to_vol_aid.get_stitch_info(os.path.join(map_path,'matl.omp2info'))
# Add image metadata in place ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
slc_to_vol_aid.add_image_info(stack_path,stack_info_0)
slc_to_vol_aid.add_image_info(map_path,map_info_0)
# ---------------------------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------------------------- 
#

#%% ORIGINAl
# Create minimal info dict --------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------- 
whole_min_keys_l = ['stage_rng','stage_phys_dims','pxl_phys_dims']
groups_min_keys_l = ['coordinates','areaInfo','enable','imgSpecs']    
stack_inf,map_inf = {'groups':{}},{'groups':{}}

for s_whole_k in whole_min_keys_l:
    stack_inf[s_whole_k] =  copy.deepcopy(stack_info_0[s_whole_k])
    map_inf[s_whole_k] = copy.deepcopy(map_info_0[s_whole_k])

for s_stck_grp in stack_info_0['groups'].keys():
    stack_inf['groups'][s_stck_grp] = {}
    for s_grp_k in groups_min_keys_l:
        if s_grp_k in stack_info_0['groups'][s_stck_grp].keys():
            stack_inf['groups'][s_stck_grp][s_grp_k] = copy.deepcopy(stack_info_0['groups'][s_stck_grp][s_grp_k])

for s_map_grp in map_info_0['groups'].keys():
    map_inf['groups'][s_map_grp] = {}
    for s_grp_k in groups_min_keys_l:
        if s_grp_k in map_info_0['groups'][s_map_grp].keys():
            map_inf['groups'][s_map_grp][s_grp_k] = copy.deepcopy(map_info_0['groups'][s_map_grp][s_grp_k])

# ----------------------------------------------------------------------------------------------------           
with open(os.path.join(save_path,'zoomed_info.json'),'w') as stack_inf_json:
    json.dump(stack_inf,stack_inf_json)
with open(os.path.join(save_path,'map_info.json'),'w') as map_inf_json:
    json.dump(map_inf,map_inf_json)





#%% COMMON
# Load files specifications ---------------------------------------------------------------------------    
stack_inf = slc_to_vol.load_json_info('zoomed_info.json',save_path)
map_inf = slc_to_vol.load_json_info('map_info.json',save_path)

#%% COMMON 
stg_um,track_bnds_d,map_idxs, stack_idxs = slc_to_vol.map_global_layout(map_inf,stack_inf)
#%% COMMON 
# Define anterior to posterior order from display
map_seq_ids = [6,7,8,9,13,12,11,10,14,15,16,17,21,20,19,18,22,23,24,25,29,28,27,26]
stck_seq_ids = [30,31,32,33,37,36,35,34,38,39,40,41,45,44,43,42,46,47,48,49,53,52,51,50]
#%% ORIGINAL 
# Section for loading and saving full array of zoomed in images as memmap --------------------------
imgs_stck_full_0 = slc_to_vol_aid.get_stack_imgs_orig(stack_path,stack_info_0,stack_idxs,stck_seq_ids, mode = 'full')
imgs_stck_full = imgs_stck_full_0[:,[0,2],...].copy() # This line is used to subset channels
# Save as memory mapped file due to huge file size
imgs_stck_full_save = memmap(os.path.join(save_path,'imgs_stck_red.tif'),shape=imgs_stck_full.shape,dtype=imgs_stck_full.dtype)
imgs_stck_full_save[...] = imgs_stck_full; imgs_stck_full_save.flush()
del(imgs_stck_full_0); del(imgs_stck_full) # Free up memory after file is saved as memmap
# ------------------------------------------------------------------------------------------------------

#%% ORIGINAL
imgs_map,map_proj_msk,map_proj_pxl_size = slc_to_vol_aid.get_map_imgs_orig(map_path,map_info_0,map_idxs,map_seq_ids,track_bnds_d,stg_um)
#%% Optional section to save map imgs-------------------------------------------------------------------
imgs_map_save = memmap(os.path.join(save_path,'img_map_single.tif'),shape=imgs_map.shape,dtype=imgs_map.dtype)
imgs_map_save[...] = imgs_map; imgs_map_save.flush()
del(imgs_map)
# ------------------------------------------------------------------------------------------------------



#%% COMMON
imgs_map = memmap(os.path.join(save_path,'img_map_single.tif'))

#%% COMMON
map_proj_msk,map_proj_pxl_size = slc_to_vol.get_map_proj(imgs_map,map_inf,map_idxs,map_seq_ids,track_bnds_d,stg_um)


#%% COMMON
slc_to_vol.plot_map_imgs(map_proj_msk,stack_inf,stack_idxs,map_proj_pxl_size,track_bnds_d,stg_um)

#%% ORIGINAL
imgs_stck = slc_to_vol_aid.get_stack_imgs_orig(stack_path,stack_info_0,stack_idxs,stck_seq_ids, mode = 'single')
# ------------------------------------------------------------------------------------------------------

#%% COMMON
imgs_stck_red = memmap(os.path.join(save_path,'imgs_stck_red.tif'))
imgs_stck = np.mean(imgs_stck_red[:,0,...],axis=1)


#%% COMMON
# ---------------------------------------------------------------------------------------------------- 
flip_trk,flip_cc_d,flip_imgs = slc_to_vol.detect_flip(imgs_stck)
#%% COMMON
imgs_stck_flp,imgs_map_flp  = slc_to_vol.correct_flip(flip_trk,imgs_stck,imgs_map)
#%% COMMON 
# SECTION TO DISPLAY IMAGES AND OUTLINE SLICES -----------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# Display images in napari to outline shapes --------------------
n_vwr = napari.Viewer() 
img_lyr = n_vwr.add_image(imgs_map_flp)
#%%
n_vwr.close()
#%% Retrieve coordinates of masks
map_clean_msks = {}
for e_s_map_idx in range(len(imgs_map_flp)):    
    try:
        s_cln_msk = n_vwr.layers['slice_'+str(e_s_map_idx)].data
        assert len(s_cln_msk)==1 and np.all(s_cln_msk[0][:,0]==e_s_map_idx), 'Error found on mask'
        map_clean_msks['slice_'+str(e_s_map_idx)] = s_cln_msk[0][:,1:].copy()

    except AssertionError as mssg:
        print(mssg)

# Save coordinates as json ------------------------------------
map_clean_msks_json = {k:map_clean_msks[k].tolist() for k in map_clean_msks.keys()}        
with open(os.path.join(save_path,'clean_msks_coords.json'),'w') as clean_msks_out:
    json.dump(map_clean_msks_json,clean_msks_out)
# --------------------------------------------------------------------------------------------------------------------
#%% --------------------------------------------------------------------------------------------------------------------



#%% COMMON 
# Load outline coordinates -------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
with open(os.path.join(save_path,'clean_msks_coords.json'),'r') as json_file_in:    
    msks_crds = json.load(json_file_in)
# --------------------------------------------------------------------------------------------------------------------

#%% COMMON 
# --------------------------------------------------------------------------------------------------------------------
imgs_map_msk,imgs_map_msk_eq,nvwr_0 = slc_to_vol.get_masked_map(imgs_map_flp,msks_crds)

#%% COMMON
# --------------------------------------------------------------------------------------------------------------------
imgs_map_reg,imgs_map_reg_eq,warp_map =  slc_to_vol.register_map_imgs(imgs_map_msk)

#%% COMMON
zoomed_transformed_d = slc_to_vol.zoomed_apply_transform(map_seq_ids,stck_seq_ids,map_inf,stack_inf,warp_map,stg_um,flip_trk,imgs_map_msk)

#%% COMMON
int_bbox,msks_int_cntr = slc_to_vol.get_zoomed_intersect(zoomed_transformed_d)

#%% ------------------------------------------------------------------------------------------------------------------

#%% Visualized transformed masks on registered map images -----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------    
mov_fps = 30
n_vwr_2 = napari.Viewer() 
animation = Animation(n_vwr_2)
n_vwr_2.add_image(np.array(zoomed_transformed_d['transformed']['img']),name='map_t',scale = [map_inf['pxl_phys_dims'],map_inf['pxl_phys_dims']])
n_vwr_2.camera.center = (0,4358,5594)
n_vwr_2.camera.zoom = 0.092

mt_p_0 = np.percentile(n_vwr_2.layers['map_t'].data,0.01)
mt_p_1 = np.percentile(n_vwr_2.layers['map_t'].data,99.99)
n_vwr_2.layers['map_t'].contrast_limits = (mt_p_0,mt_p_1)
n_vwr_2.layers['map_t'].colormap = 'gray_r' 

n_vwr_2.scale_bar.visible = True
n_vwr_2.scale_bar.unit = "um"
n_vwr_2.scale_bar.font_size = 20
n_vwr_2.scale_bar.ticks = False
n_vwr_2.scale_bar.colored = True
n_vwr_2.scale_bar.color = 'darkorange'

for id_crd_set,s_crd_set in enumerate(zoomed_transformed_d['transformed']['edgeCrds']):
    modif_crd_set = np.append(np.ones((1,4))*id_crd_set,s_crd_set*map_inf['pxl_phys_dims'],axis=0)
    n_vwr_2.add_shapes(data = modif_crd_set.T,name='msk_'+str(id_crd_set),shape_type='polygon', face_color='orange',edge_color='darkorange',opacity=0.25,visible=False)    

n_vwr_2.dims.current_step = (0,0,0)
animation.capture_keyframe(1)
animation.capture_keyframe(steps=int(0.5*mov_fps))
n_vwr_2.dims.current_step = (23,0,0)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(0.5*mov_fps))
for i in range(imgs_map_reg.shape[0]): n_vwr_2.layers['msk_'+str(i)].visible = True
animation.capture_keyframe(steps=int(1.5*mov_fps))
n_vwr_2.dims.current_step = (0,0,0)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=4*mov_fps)
#%%
animation.animate(save_path/'test_mov_2.mp4',fps=mov_fps,quality=9,canvas_only=True)
#%%
n_vwr_2.add_shapes(data = int_bbox*map_inf['pxl_phys_dims'],shape_type='rectangle', face_color='red',opacity=0.5)    
n_vwr_2.add_shapes(data = msks_int_cntr.T*map_inf['pxl_phys_dims'],shape_type='polygon', face_color='green',opacity=0.5)    

# ----------------------------------------------------------------------------------------------------------------------    


#%%



#%% COMMON
# ----------------------------------------------------------------------------------------------------------------------    
slc_to_vol.align_zoomed_in(int_bbox,map_inf,stack_inf,map_seq_ids,stck_seq_ids,imgs_stck_red,zoomed_transformed_d,save_path,flip_trk)

#%% COMMON
# -------------------------------------------------------------------------------------------------------------------------------------    
slc_to_vol.zoomed_in_dims_match(save_path)


#%% COMMON
# Load zoomed aligned data set -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------    
zommed_align_mmap  = memmap(os.path.join(save_path,'zommed_align_matched.tif'))
#%% COMMON
# -------------------------------------------------------------------------------------------------------------------------------------    
nw_0 = slc_to_vol.visualize_stcks_vol(zommed_align_mmap,stack_inf,stck_seq_ids,z_offset=80,display_mode='zProj_block',skip_idx = [17])
#%% MOVIE DOR VOL DISPLAY ----------------------------------------------------------------------------------------------------------------------
mov_fps = 30 
animation = Animation(nw_0)
# --------------------------------------------------------------------------------------------------------------------------
nw_0.camera.center = (1083,1075,1259)
nw_0.camera.zoom = 0.32
nw_0.camera.angles = (-2,12,80)
nw_0.camera.perspective = 5
nw_0.scale_bar.visible = True
nw_0.scale_bar.unit = "um"
nw_0.scale_bar.font_size = 20
nw_0.scale_bar.ticks = False
nw_0.scale_bar.colored = True
nw_0.scale_bar.color = 'royalblue'
nw_0.layers['blue'].gamma = 1.25
nw_0.layers['green'].gamma = 1.5
nw_0.layers['red'].gamma = 1.5
nw_0.layers['red'].contrast_limits = (0,0.060)
animation.capture_keyframe(1)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (4,-22,91)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.layers['blue'].visible = False
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.center = (1277,729,763)
nw_0.camera.zoom = 0.91
animation.capture_keyframe(steps=3*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (7,-24,85)
animation.capture_keyframe(steps=mov_fps)
nw_0.camera.angles = (11,8,61)
animation.capture_keyframe(steps=5*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (0,-38,103)
animation.capture_keyframe(steps=5*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (7,-24,85)
animation.capture_keyframe(steps=3*mov_fps)
nw_0.camera.angles = (4,-22,91)
animation.capture_keyframe(steps=3*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.zoom = 0.32
nw_0.camera.center = (1083,1075,1259)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.layers['blue'].visible = True
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (-2,12,80)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (4,12,99)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
nw_0.camera.angles = (-2,12,80)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=4*mov_fps)
#%%
animation.animate(save_path/'test_mov_4.mp4',fps=mov_fps,quality=9,canvas_only=True)


#%% MOVIE FOR Z-PROJ RAW -------------------------------------------------------------------------------------------------------------- 
# --------------------------------------------------------------------------------------------------------------------------
mov_fps = 30 
animation = Animation(nw_0)
nw_0.layers['blue'].contrast_limits = (0,0.061)
nw_0.layers['blue'].gamma = 1.1
nw_0.layers['green'].gamma = 1.6
nw_0.layers['green'].contrast_limits = (0,0.06)
nw_0.layers['red'].gamma =1.2
nw_0.layers['red'].contrast_limits = (0,0.03)
nw_0.scale_bar.visible = True
nw_0.scale_bar.unit = "um"
nw_0.scale_bar.font_size = 20
nw_0.scale_bar.ticks = False
nw_0.scale_bar.colored = True
nw_0.scale_bar.color = 'royalblue'
nw_0.camera.center = (0,1078,1405)
nw_0.camera.zoom = 0.39
nw_0.dims.current_step = (nw_0.layers['blue'].data.shape[0],0,0)
animation.capture_keyframe(1)
animation.capture_keyframe(steps=int(0.5*mov_fps))
nw_0.dims.current_step = (0,0,0)
animation.capture_keyframe(steps=5*mov_fps)
animation.capture_keyframe(steps=5*mov_fps)

#%%
animation.animate(save_path/'test_mov_3.mp4',fps=mov_fps,quality=9,canvas_only=True)








#%% COMMON 
imgs_stck_full = memmap(os.path.join(save_path,'imgs_stck_red.tif'))
# SECTION FOR MOVIES CREATION  --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------



#%%
#%% MOVIE FOR GLOBAL LAYOUT VISUALIZATION ------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
n = slc_to_vol_aid.visualize_layout(map_proj_msk,imgs_stck_full,map_proj_pxl_size,stack_inf,stck_seq_ids,stg_um,track_bnds_d,mode = 'full')
mov_fps = 30
#%% Histology layout exploration -----------------------------------------------------------------------------------
# Adjust channels config 
n= slc_to_vol_aid.adjust_zoomed('colormap_all',['blue'],n,stck_seq_ids,'bop blue') # match zoomed in blue color to whole projection
n= slc_to_vol_aid.adjust_zoomed('contrast_all',['blue'],n,stck_seq_ids,[0.01,99.99]) 
n= slc_to_vol_aid.adjust_zoomed('gamma_all',['blue'],n,stck_seq_ids,1) 
n= slc_to_vol_aid.adjust_zoomed('contrast_all',['green'],n,stck_seq_ids,[0.01,99.9]) 
n= slc_to_vol_aid.adjust_zoomed('gamma_all',['green'],n,stck_seq_ids,2) 
n= slc_to_vol_aid.adjust_zoomed('contrast_all',['red'],n,stck_seq_ids,[0.01,99.9]) 
n= slc_to_vol_aid.adjust_zoomed('gamma_all',['red'],n,stck_seq_ids,1.2) 
#%%
animation = Animation(n)
# 1 - Global display with only blue channel
n.camera.center = (0,22336,25918) # Value taken directly from vis
n.camera.zoom = 0.0217 # Value taken directly from vis
n= slc_to_vol_aid.adjust_zoomed('hide_all',['blue','green','red'],n,stck_seq_ids,None) # Turn off all zoomed in channels
n.layers['whole_proj'].colormap = 'bop blue' # Set whole projection colormap
# Set colormap to percentiles 
wp_p_0 = np.percentile(n.layers['whole_proj'].data,0.01)
wp_p_1 = np.percentile(n.layers['whole_proj'].data,99.99)
n.layers['whole_proj'].contrast_limits = (wp_p_0,wp_p_1)
n.layers['whole_proj'].gamma = 1.2
n.scale_bar.visible = True
n.scale_bar.unit = "um"
n.scale_bar.font_size = 20
n.scale_bar.ticks = False
n.scale_bar.colored = True
n.scale_bar.color = 'darkorange'
animation.capture_keyframe(steps=1)
animation.capture_keyframe(steps=3*mov_fps)
# 2 - Zoom into slice 40, display high res blue channel and then red/green
n.camera.center = (0,18091,30753)
n.camera.zoom = 0.472
animation.capture_keyframe(steps=5*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
n= slc_to_vol_aid.adjust_zoomed('show_40',['blue'],n,stck_seq_ids,None) # Toggle blue zoomed in for 39
animation.capture_keyframe(steps=int(1.5*mov_fps))
animation.capture_keyframe(steps=int(1.5*mov_fps))
n= slc_to_vol_aid.adjust_zoomed('show_40',['red','green'],n,stck_seq_ids,None) # Toggle red/green zoomed in for 39
animation.capture_keyframe(steps=int(1.5*mov_fps))
# 3 - Hide blue channel, zoom in even more and show red gren alone
n.layers['whole_proj'].visible = False
n= slc_to_vol_aid.adjust_zoomed('hide_40',['blue'],n,stck_seq_ids,None) # Toggle blue zoomed in for 39
animation.capture_keyframe(steps=int(1.5*mov_fps))
n.camera.zoom = 2.05
n.camera.center = (0,18852,30557)
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
n.camera.center = (0,18290,30050)
animation.capture_keyframe(steps=3*mov_fps)
n.camera.center = (0,17895,29808)
animation.capture_keyframe(steps=3*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
n.camera.center = (0,18091,30753)
n.camera.zoom = 0.472
animation.capture_keyframe(steps=4*mov_fps)
animation.capture_keyframe(steps=int(1.5*mov_fps))
# 4 - Zoom out again and turn on whole_proj plus all zoomed in channels
n.layers['whole_proj'].visible = True
n= slc_to_vol_aid.adjust_zoomed('show_all',['blue','red','green'],n,stck_seq_ids,None) # Toggle blue zoomed in for 39
animation.capture_keyframe(steps=int(1.5*mov_fps))
# 5 - Adjust blue/red/green for slice 40 and move there
n.camera.center = (0,18460,37716)
animation.capture_keyframe(steps=3*mov_fps)
animation.capture_keyframe(steps=3*mov_fps)
n.camera.center = (0,7320,37432)
animation.capture_keyframe(steps=3*mov_fps)
animation.capture_keyframe(steps=3*mov_fps)
# 6 - Zoom out and turn on whole_proj
n= slc_to_vol_aid.adjust_zoomed('hide_47',['blue','red','green'],n,stck_seq_ids,None) # Toggle blue zoomed in for 39
n.camera.center = (0,22336,25918) # Value taken directly from vis
n.camera.zoom = 0.0217 # Value taken directly from vis
animation.capture_keyframe(steps=8*mov_fps)
animation.capture_keyframe(steps=12*mov_fps)
#%%
animation.animate(save_path/'test_mov_0.mp4',fps=mov_fps,quality=9,canvas_only=True)

#%% Line to get display back to original of video
n= slc_to_vol_aid.adjust_zoomed('hide_all',['blue','red','green'],n,stck_seq_ids,None) # Toggle blue zoomed in for 39
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------




#%% MOVIE FOR REGISTRATION OF MAP IMAGES ------------------------------------------------------------------------------------------------------------------
mov_fps = 30
n_vwr_8 = napari.Viewer()
animation = Animation(n_vwr_8)
n_vwr_8.add_image(map_proj_msk,scale = [map_inf['pxl_phys_dims'],map_inf['pxl_phys_dims']])
mp_p_0 = np.percentile(n_vwr_8.layers['map_proj_msk'].data,0.01)
mp_p_1 = np.percentile(n_vwr_8.layers['map_proj_msk'].data,99.99)
n_vwr_8.layers['map_proj_msk'].contrast_limits = (mp_p_0,mp_p_1)
n_vwr_8.layers['map_proj_msk'].colormap = 'gray_r' 
n_vwr_8.camera.center = (0,22336,25918) # Value taken directly from vis
n_vwr_8.camera.zoom = 0.0217 # Value taken directly from vis
n_vwr_8.scale_bar.visible = True
n_vwr_8.scale_bar.unit = "um"
n_vwr_8.scale_bar.font_size = 20
n_vwr_8.scale_bar.ticks = False
n_vwr_8.scale_bar.colored = True
n_vwr_8.scale_bar.color = 'darkorange'
n_vwr_8.camera.center = (0,22312,25305)
n_vwr_8.camera.zoom = 0.0214

for s_stck in stck_seq_ids:
    # Get square patches in microns ----------------------------------------------------------
    s_stck_v0_um = stack_inf['groups'][s_stck]['coordinates']['y']*stg_um-track_bnds_d['min_v']
    s_stck_x0_um = stack_inf['groups'][s_stck]['coordinates']['x']*stg_um-track_bnds_d['min_x']
    s_stck_v_ext = stack_inf['groups'][s_stck]['coordinates']['height']*stg_um
    s_stck_x_ext = stack_inf['groups'][s_stck]['coordinates']['width']*stg_um
    s_stck_v1_um = s_stck_v0_um + s_stck_v_ext
    s_stck_x1_um = s_stck_x0_um + s_stck_x_ext
    # Manually arrange coordinates in circular path for patch representation
    s_stck_crds = np.array([[s_stck_v0_um,s_stck_x0_um],[s_stck_v1_um,s_stck_x1_um]])
    n_vwr_8.add_shapes(data = s_stck_crds,name='zoomed_'+str(s_stck),shape_type='rectangle', face_color='orange',edge_color='darkorange',opacity=0.25,visible=False)

animation.capture_keyframe(steps=1)
animation.capture_keyframe(steps=2*mov_fps)
for s_stck in stck_seq_ids: n_vwr_8.layers['zoomed_'+str(s_stck)].visible = True
animation.capture_keyframe(steps=3*mov_fps)

animation.animate(save_path/'test_mov_1.mp4',fps=mov_fps,quality=9,canvas_only=True)

#%% MOVIES FOR REGISTRATION IN ZOOMED IN IMAGES





#%%
    stg_um,
    track_bnds_d
map_proj_pxl_size
#%%
        for s_map,s_map_img_float in zip(reversed(map_ant_to_post_ids),np.flip(map_imgs,axis=0)):
        # Reversed order is to ensure overlapping between map images not happenning in region of interest    
            # Downscale image  ----------------------------------------------------------------
            # NOTE: Images in input array are already sorted for this function 
            s_map_img = downscale_local_mean(s_map_img_float,(down_sample,down_sample))
            # Calculate image offset in pixels -------------------------------------------------------
            s_map_v0 = (map_info['groups'][s_map]['coordinates']['y']*stg_to_um-track_bounds_d['min_v'])/mapProj_pxl_size
            s_map_x0 = (map_info['groups'][s_map]['coordinates']['x']*stg_to_um-track_bounds_d['min_x'])/mapProj_pxl_size
            s_map_crds = rectangle((s_map_v0,s_map_x0),extent=s_map_img.shape)


















#%% EXTRA DOCUMEMTATION ------------------------------------------------------------------------
from skimage.draw import line, rectangle, polygon, polygon2mask, rectangle_perimeter
from scipy.ndimage import gaussian_filter
from skimage import  measure
from skimage.filters import threshold_otsu
from skimage.segmentation import flood



# TO KEEP DEVELOPING: Extract slices boundaries -----------------------------------------------
# Needs a bit more of work but it is kind of going --------------------------
# IDEA: Process in patches finding a different otsu threshold for each part

reg_map_eq_extr = []

for idx,e_reg_map_eq in enumerate(reg_map_eq):
    gauss_filt_img = gaussian_filter(e_reg_map_eq,sigma=100)
    cntrs_th = threshold_otsu(gauss_filt_img) 
    cntrs = measure.find_contours(gauss_filt_img,cntrs_th)
    cntr_temp_img = np.zeros_like(gauss_filt_img)
    for e_cntr in cntrs: cntr_temp_img[np.round(e_cntr[:,0]).astype('int'),\
                                       np.round(e_cntr[:,1]).astype('int')] = 1
    center_point = (np.array(cntr_temp_img.shape)/2).astype('int')   
    extr_msk = flood(cntr_temp_img,tuple(center_point),connectivity=1) 
    reg_map_eq_extr.append(e_reg_map_eq*extr_msk)  
    print(f'Done with image : {idx}')

reg_map_eq_extr = np.array(reg_map_eq_extr)

#%% ---------------------------------------------------------------------------------------------------------
#%% TESTS TO SAVE ----------------------------------------------------------------------------------------------------------------------
# Applying mask when doing opencv registration 
d = polygon2mask(map_imgs[0].shape,c)
d_dwn = downscale_local_mean(d,(reg_dwn_factor,reg_dwn_factor))
d_dwn_ub = img_as_ubyte(d_dwn)
d_dwn_ub_int = d_dwn.astype(np.uint8)

cc_test = []
w_mat_test = []
for test_spec in [None,d_dwn_ub,d_dwn_ub_int,d_dwn_ub_int+1]:
    map_ref = map_imgs[0]
    map_reg = map_imgs[1]
    map_ref_dwn = downscale_local_mean(map_ref,(reg_dwn_factor,reg_dwn_factor))
    map_reg_dwn = downscale_local_mean(map_reg,(reg_dwn_factor,reg_dwn_factor))
    warp_mode = cv2.MOTION_AFFINE #Select transformation mode
    number_of_iterations = 1000;# Specify the number of iterations.
    termination_eps = 1e-10;# Specify the threshold of the increment in the correlation coefficient between two iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)# Define termination criteria
    warp_matrix = np.eye(2, 3, dtype=np.float32) 
    (cc, warp_matrix) = cv2.findTransformECC (map_ref_dwn,map_reg_dwn,warp_matrix, warp_mode, criteria,test_spec,5)
    cc_test.append(cc)
    w_mat_test.append(warp_matrix)
