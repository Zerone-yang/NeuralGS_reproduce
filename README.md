1. 准备3dgs训练后的ply文件和camera.json文件
2. main.py用于对ply文件压缩，会输出压缩后的模型和存储高斯中心点的ply文件
3. test.py用于给定camera.json和ply输出渲染图片（目只渲染第一张），并和原图做psnr计算
4. output_ply.py用于将压缩后的模型和存储高斯中心点的ply文件重新解压为原始ply文件
5. importance.py用于计算重要性
6. models.py模型结构和训练代码
