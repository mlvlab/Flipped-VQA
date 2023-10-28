import torch
from .base_dataset import BaseDataset
import json

class DramaQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/dramaqa/AnotherMissOhQA_{split}_set.json', "r"))
        self.features = torch.load(f'./data/dramaqa/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data[idx]["que"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = self.data[idx]['answers']

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video_id , idx):
        
        scene = True
        # Scene
        if video_id[-4:] == '0000':
            shots = self.data[idx]['shot_contained']
            start, end = shots[0], shots[1]

            for i in range(start, end+1):
                v_name = video_id[:-4] + f'{i:04}'

                if v_name not in self.features.keys(): 
                    print(v_name, " Not in features")
                    nxt_vid = torch.zeros(1, self.features_dim)
                else: nxt_vid = self.features[v_name].float()

                if i == start: video = nxt_vid
                else: video = torch.concat((video, nxt_vid), dim = 0)
        # Shot
        else:
            scene = False
            if video_id not in self.features.keys():
                print(video_id, "Not in freatures")
                video = torch.zeros(1, self.features_dim)
            else:
                video = self.features[video_id].float()

        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len, scene

    def __getitem__(self, idx):
        vid = self.data[idx]['vid']
        qtype = -1
        answer = self.data[idx]['correct_idx']
        text = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        video, video_len, scene = self._get_video(f'{vid}', idx)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}

    def __len__(self):
        return len(self.data)