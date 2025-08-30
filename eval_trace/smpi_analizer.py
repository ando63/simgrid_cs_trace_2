import pandas as pd
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt

def process_csv_to_matrix_txt(input_csv_path, output_txt_path, output_txt_path_2, matrix_size=64):
    global simulation_times_full_fat, simulation_times_half_fat
    """
    CSVファイルからタイムスタンプを除いた行列データを抽出し、
    指定された形式でテキストファイルに保存します。

    Args:
        input_csv_path (str): 入力CSVファイルのパス。
        output_txt_path (str): 出力テキストファイルのパス。
        matrix_size (int): 行列の一辺のサイズ (例: 4x4の場合は4)。
    """
    try:
        # CSVファイルを読み込む。最初の行をヘッダーとして解釈
        # df = pd.read_csv(input_csv_path)
        # df = pd.read_csv(input_csv_path, header=None, sep=',')
        df = pd.read_csv(input_csv_path, header=None, sep=',', skiprows=1)

        # 最初の列（タイムスタンプ）を削除
        df = df.iloc[:, 1:]
        print(df)
        simulation_times_full_fat = []
        simulation_times_half_fat = []

        # 出力ファイルを開き、データを書き込む
        
        with open(output_txt_path, 'w') as f:
            with open(output_txt_path_2, 'w') as f2:
                # 1行ずつ行列データを取り出す
                for _, row in df.iterrows():
                    print(row)
                    # 行のデータをNumPy配列に変換
    
                    first_16_elements = row[:16]
                    remaining_elements = row[16:]
                    matrix_data_4_4 = np.array(first_16_elements.values)
                    matrix_data_4_4[matrix_data_4_4 < 0] = 0
    
                    matrix_data_64_64 = np.array(remaining_elements.values)
                    matrix_data_64_64[matrix_data_64_64 < 0] = 0
    
                    # 1次元配列を行列の形（例: 4x4）にreshape
                    reshaped_matrix_4_4 = matrix_data_4_4.reshape(4, 4)
                    reshaped_matrix_64_64 = matrix_data_64_64.reshape(64,64)
    
                    # 行列データをファイルに書き込む
                    for i in range(4):
                        # 各行の要素をスペース区切りで文字列化
                        row_str = ' '.join(map(str, reshaped_matrix_4_4[i]))
                        f.write(row_str + '\n')
    
                    for i in range(64):
                        row_str = ' '.join(map(str, reshaped_matrix_64_64[i]))
                        f2.write(row_str + '\n')
                    
                    # 各行列の間に空行を挿入して見やすくする
                    f.write('\n')
                    f2.write('\n')
                    f.close()
                    f2.close()
    
                    # gen_traf
    
                    subprocess.run(
                        ["python3", "gen_traffic.py", "matrix_2.txt", "1", "traf_matrix/"],
                        check=True
                    )
    
                    # smpirun(with full.xml)
    
                    cmd = "smpirun -no-privatize -replay traf_matrix/matrix_2.txt_1 --log=replay.thresh:critical --log=no_loc -np 64 -platform xmlfiles/link_config.yaml_fat_treemesh_cs_file.xml ./replay/smpi_replay --log=smpi_config.thres:warning --log=xbt_cfg.thres:warning > Output.txt 2>&1"
    
                    subprocess.run(cmd, shell=True, check=True)
    
                    # 結果append
    
                    with open("Output.txt", "r") as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        # "Simulation time <数字>" を探す
                        match = re.search(r"Simulation time (\d+\.\d+)", line)
                        if match:
                            # 数字部分を float に変換してリストに追加
                            simulation_times_full_fat.append(float(match.group(1)))
    
    
    
                    # smpirun(with half.xml)
    
                    cmd = "smpirun -no-privatize -replay traf_matrix/matrix_2.txt_1 --log=replay.thresh:critical --log=no_loc -np 64 -platform xmlfiles/link_config.yaml_half_fat_treemesh_cs_file.xml ./replay/smpi_replay --log=smpi_config.thres:warning --log=xbt_cfg.thres:warning > Output_2.txt 2>&1"
    
                    subprocess.run(cmd, shell=True, check=True)
    
                    # 結果append
    
                    with open("Output_2.txt", "r") as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        # "Simulation time <数字>" を探す
                        match = re.search(r"Simulation time (\d+\.\d+)", line)
                        if match:
                            # 数字部分を float に変換してリストに追加
                            simulation_times_half_fat.append(float(match.group(1)))

                    if _ == 235:
                        exit()
    
                    # f.close()  # 一旦閉じる
                    open(output_txt_path, 'w').close()  # 中身を空にする
                    open(output_txt_path_2, 'w').close()
    
                    # 🔽 また追記モードで開き直す
                    f = open(output_txt_path, 'a')
                    f2 = open(output_txt_path_2, 'a')
            
    
            print(f"データの処理が完了しました。ファイル '{output_txt_path}' に保存されました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_csv_path}' が見つかりませんでした。パスを確認してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# スクリプトの実行例
if __name__ == "__main__":
    global simulation_times_full_fat, simulation_times_half_fat
    # CSVファイルのパス
    csv_file_path = "time_traffic_matrix.csv"
    # 出力テキストファイルのパス
    txt_file_path = "traf_matrix/matrix.txt"
    txt_file_path_2 = "traf_matrix/matrix_2.txt"
    
    # 実際に行列のサイズに合わせて変更してください
    # 例：4x4行列の場合は4
    matrix_dimension = 64 
    print(matrix_dimension)

    process_csv_to_matrix_txt(csv_file_path, txt_file_path, txt_file_path_2, matrix_dimension)

    print(simulation_times_full_fat)
    plt.figure(figsize=(12, 4))
    # plt.plot(simulation_times_full_fat, marker='o', markersize=3, linestyle='-')
    plt.plot(simulation_times_full_fat, label="Full Fat", marker='o', markersize=3)

    # half_fat のプロット
    plt.plot(simulation_times_half_fat, label="Half Fat", marker='x', markersize=3)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Array values from left to right')
    plt.grid(True)
    
    # 画像として保存
    plt.savefig("array_plot.png", dpi=300)  # dpi は解像度、必要に応じて調整
    plt.close()  # メモリ解放のために閉じる
