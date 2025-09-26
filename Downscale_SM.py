import os
import numpy as np
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
import rasterio
from tqdm import tqdm

# 输入和输出路径
coarse_sm_folder = r""
fine_awc_path = r''
awc_mean_path = r''
awc_std_path = r""
fine_sm_std_path = r""
output_folder = r""

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)


# 土壤水分方程
def soil_moisture_variance(theta_mean, theta_std, proxy_std, proxy_mean, proxy_fine):
    z_scores = (proxy_fine - proxy_mean) / proxy_std
    downscaled = theta_mean + theta_std * z_scores
    return downscaled.astype(np.float32)


def gradient_interpolation(coarse_data, fine_proxy, block_size=256, gaussian_sigma=1.0):
    smoothed_data = coarse_data.copy().astype(np.float32)
    height, width = fine_proxy.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)

            block_coarse = coarse_data[y:y_end, x:x_end]
            block_fine_proxy = fine_proxy[y:y_end, x:x_end]

            if block_fine_proxy.size > 0:
                valid_mask = ~np.isnan(block_coarse)

                if np.any(valid_mask):
                    local_mean = np.nanmean(block_coarse[valid_mask])
                    local_std = np.nanstd(block_coarse[valid_mask])
                    gradient_x, gradient_y = np.gradient(block_fine_proxy.astype(np.float32))
                    smoothed_gradient_x = gaussian_filter(gradient_x, sigma=gaussian_sigma)
                    smoothed_gradient_y = gaussian_filter(gradient_y, sigma=gaussian_sigma)

                    smoothed_block = block_coarse.copy()
                    adjustment = (
                            smoothed_gradient_x * local_std +
                            smoothed_gradient_y * local_std
                    )
                    smoothed_block[valid_mask] += adjustment[valid_mask]
                    smoothed_data[y:y_end, x:x_end] = smoothed_block

    smoothed_data = gaussian_filter(smoothed_data, sigma=0.5)
    return smoothed_data


def align_raster(input_path, reference_path):
    with rasterio.open(input_path) as src, rasterio.open(reference_path) as ref:
        aligned_data = np.empty((ref.height, ref.width), dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=aligned_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.bilinear
        )
        return aligned_data


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
        nodata_value = profile.get('nodata', None)
        if nodata_value is not None:
            data[data == nodata_value] = np.nan
        return data, profile


def write_raster(output_path, data, profile):
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)


def rescale_soil_moisture(data):
    # 将小于0的值设为0
    data[data < 0] = 0


    return data


def downscale_soil_moisture_for_all():
    _, fine_profile = read_raster(fine_awc_path)

    awc_mean_aligned = align_raster(awc_mean_path, fine_awc_path)
    awc_std_aligned = align_raster(awc_std_path, fine_awc_path)
    fine_sm_std_aligned = align_raster(fine_sm_std_path, fine_awc_path)
    fine_awc, _ = read_raster(fine_awc_path)

    # 遍历所有土壤水分TIF文件
    tif_files = [f for f in os.listdir(coarse_sm_folder) if f.endswith('.tif')]

    for file in tqdm(tif_files, desc="处理TIF文件"):
        coarse_sm_path = os.path.join(coarse_sm_folder, file)
        output_path = os.path.join(output_folder, file)

        coarse_sm, _ = read_raster(coarse_sm_path)
        resampled_coarse_sm = align_raster(coarse_sm_path, fine_awc_path)

        downscaled_sm = soil_moisture_variance(
            resampled_coarse_sm, fine_sm_std_aligned,
            awc_std_aligned, awc_mean_aligned, fine_awc
        )

        smoothed_sm = gradient_interpolation(downscaled_sm, fine_awc)
        final_sm = smoothed_sm.copy()
        nan_mask = np.isnan(final_sm)
        final_sm[nan_mask] = downscaled_sm[nan_mask]

        # 重新缩放
        final_sm = rescale_soil_moisture(final_sm)

        write_raster(output_path, final_sm, fine_profile)


if __name__ == "__main__":
    downscale_soil_moisture_for_all()
