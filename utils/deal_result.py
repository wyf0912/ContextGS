# import glob
# import re
# result_paths = glob.glob("outputs_/final_ours/*/*/outputs.log")

# with open("results.csv", "w") as results:
#     results.write("dataset, sense, lmbda, size, psnr, ssim, lpips\n")
#     for path in result_paths:
#         ps = path.split("/")
#         dataset = ps[-3]
#         sense, lmbda = ps[-2].split("_")
#         # print(lmbda, sense, dataset)
#         with open(path, "r") as f:
#             lines = f.readlines()
#         res = list(filter(lambda line: "Estimated sizes in MB" in line or "Evaluating test" in line, lines))
#         if len(res)!=2:
#             print(path)
#             print(res)
#         else:
#             # print(res)
#             size = re.findall(r"\d*\.\d+", res[0])[-1]
#             psnr, ssim, lpips = re.findall(r"\d*\.\d+", res[1])[1:]
#             results.write(f"{dataset}, {sense}, {lmbda}, {size}, {psnr}, {ssim}, {lpips}\n")



## deal results of multi_render

# import glob
# import re
# result_paths = glob.glob("outputs/final_ours_multi_render/blending/*/outputs.log")
# print(result_paths)
# with open("results.csv", "w") as results:
#     results.write("dataset, sense, lmbda, size, psnr, , , ssim, , ,lpips, , ,\n")
#     for path in result_paths:
#         ps = path.split("/")
#         # print(ps)
#         dataset = ps[-3]
#         try:
#             sense, lmbda = ps[-2].split("_")
#         except:
#             continue
#         # print(lmbda, sense, dataset)
#         with open(path, "r") as f:
#             lines = f.readlines()
#         res = list(filter(lambda line: "Estimated sizes in MB" in line or "Evaluating test" in line, lines))
#         if len(res)!=2:
#             print(path)
#             print(res)
#         else:
#             # print(res)
#             print(res[1], path)
#             size = re.findall(r"\d*\.\d+", res[0])[-1]



#             # 正则表达式模式
#             psnr_pattern = r'PSNR \[([0-9.\-, ]+)\]'
#             ssim_pattern = r'ssim \[([0-9.\-, ]+)\]'
#             lpips_pattern = r'lpips \[([0-9.\-, ]+)\]'

#             # 提取PSNR, SSIM, LPIPS的值
#             psnr_values = re.findall(psnr_pattern, res[1])[0]
#             ssim_values = re.findall(ssim_pattern, res[1])[0]
#             lpips_values = re.findall(lpips_pattern, res[1])[0]

#             # psnr, ssim, lpips = re.findall(r"[(\d*\.\d+)+]", res[1])
#             results.write(f"{dataset}, {sense}, {lmbda}, {size}, {psnr_values}, {ssim_values}, {lpips_values}\n")




import json
import glob
import re
result_paths = glob.glob("outputs/retrain2_final_ours/mipnerf360/*/outputs.log")
# print(result_paths)
with open("results.csv", "w") as results:
    results.write("dataset, sense, lmbda, size, psnr, ssim, lpips\n")
    for path in result_paths:
        ps = path.split("/")
        dataset = ps[-3]
        sense,  lmbda = ps[-2].split("_")
        # print(lmbda, sense, dataset)
        with open(path, "r") as f:
            lines = f.readlines()
        try:
            size = list(filter(lambda line: "Encoded sizes in MB" in line, lines))[-1]
        except:
            print(path)
            continue
        # print(res)
        size = re.findall(r"\d*\.\d+", size)[-2]

        with open("/".join((path.split("/")[:-1]+["results.json"])), 'r') as res_file:
            res = json.load(res_file)
            psnr, ssim, lpips = res["ours_30000"]["PSNR"], res["ours_30000"]["SSIM"], res["ours_30000"]["LPIPS"]
        results.write(f"{dataset}, {sense}, {lmbda}, {size}, {psnr}, {ssim}, {lpips}\n")