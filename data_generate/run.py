import json
import pandas as pd
from tqdm import tqdm
from utils import get_unique_consultation_nums, create_consultation_history, preprocess_data, get_gpt_summary, postprocess_data


with open('config.json', 'r') as f:
    config_data = json.load(f)




if __name__ == "__main__":
    
    # get csv data
    filelist_df = pd.read_csv(config_data["csv_file_path"], sep=',',quotechar='"', header=0)
    # get unique consultation numbers list
    unique_consultation_nums = get_unique_consultation_nums(filelist_df)
    

    target_consultation_nums = unique_consultation_nums[-config_data["consultation_num_from"]:-config_data["consultation_num_to"]]
    
    small_df = filelist_df[(filelist_df['consultation_num'] >= target_consultation_nums[0]) & (filelist_df['consultation_num'] <= target_consultation_nums[-1])]
    res_arr = []
    for con_num in tqdm(target_consultation_nums):
        res_arr.append(create_consultation_history(small_df, con_num))
    
    res_arr_preprocessed = preprocess_data(res_arr)
    path_to_save = config_data["path_to_save"]
    
    for idx, content in tqdm(enumerate(res_arr_preprocessed)):
        try:
            _summary = get_gpt_summary(content["dialogue"])
            res_arr_preprocessed[idx]["summary"] = _summary
        except Exception as e:
            print(e)
            res_arr_preprocessed[idx]["summary"] = ""

    res_arr_postprocessed = postprocess_data(res_arr_preprocessed)

    with open(path_to_save, "w", encoding="utf-8") as file:
        json.dump(res_arr_postprocessed, file)