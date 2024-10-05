import os
from whisper.utils import get_writer
from whisper import Whisper


class WhisperTranscriber:

    def __init__(self, model: Whisper):
        self.model = model
        self.audio_dir = None
        self.audio_name = None

    def transcribe(self, audio_dir, audio_name, options=None):
        audio_path = os.path.join(audio_dir, audio_name)
        if os.path.exists(audio_path) is False:
            return None
        self.audio_dir = audio_dir
        self.audio_name = audio_name

        options_setting = {
            "language": "japanese",
            "verbose": True,
            "logprob_threshold": -1.0,
            "hallucination_silence_threshold": 5.0,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }
        if options is not None:
            options_setting.update(options)

        try:
            print(f"開始轉譯: {audio_path}")
            result = self.model.transcribe(audio_path, **options_setting)
            self._trans2srt(result)
            self._trans2txt(result)
        except Exception as e:
            print(e)

    def _trans2srt(self, result):
        """
        將轉錄結果轉換為 SRT 檔案，並保存在與原始音訊檔案相同的目錄中。
        Args:
            result (dict): 來自 Whisper 模型的轉錄結果。
        Raises:
            Exception: 如果在寫入過程中發生錯誤。
        """
        # 產生輸出的 .srt 檔案名稱，並放置於與原檔案相同的目錄
        srt_file_name = f"{os.path.splitext(self.audio_name)[0]}.srt"
        output_srt_path = os.path.join(self.audio_dir, srt_file_name)

        try:
            # 初始化 whisper 的 writer，這裡設置為 'srt' 格式
            writer = get_writer("srt", self.audio_dir)
            # 將結果寫入 .srt 檔案
            writer(result, output_srt_path)

            print(f"轉譯完成並儲存為: {output_srt_path}")
        except Exception as e:
            print(e)

    def _trans2txt(self, result):
        """
        將轉錄結果段落轉換為帶有時間戳的格式化文本檔案。
        Args:
            result (dict): 包含轉錄結果和段落的字典。
        Returns:
            None: 該函數將格式化的轉錄結果寫入與音訊檔案相同目錄中的 .txt 檔案。
        """
        segments = result["segments"]
        # 轉換格式
        timestamps_and_text = []
        for item in segments:
            start = self._format_time(item["start"])
            end = self._format_time(item["end"])
            text = item["text"]
            timestamps_and_text.append(f"[{start} --> {end}] {text}")

        # 輸出結果
        # for line in timestamps_and_text:
        #     print(line)

        # 產生輸出的 .txt 檔案名稱，並放置於與原檔案相同的目錄
        txt_file_name = f"{os.path.splitext(self.audio_name)[0]}.txt"
        output_txt_path = os.path.join(self.audio_dir, txt_file_name)
        # 將結果輸出為 txt 檔案
        with open(output_txt_path, "w", encoding="utf-8") as file:
            for line in timestamps_and_text:
                file.write(line + "\n")

        print(f"檔案已成功輸出為 {txt_file_name}")

    def _format_time(self, seconds: float) -> str:
        """
        將以秒為單位的時間持續時間格式化為 HH:MM:SS.mmm 或 MM:SS.mmm 格式的字串。

        Args:
            seconds (float): 以秒為單位的時間持續時間。
        Returns:
            str: 格式化的時間字符串。
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:06.3f}"
        else:
            return f"{minutes:02}:{seconds:06.3f}"
