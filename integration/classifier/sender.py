# from pyjarowinkler import distance as jaro
from fuzzywuzzy import fuzz

judicial_senders = {
  "Следственный комитет Российской Федерации": [
    "Следственный комитет Российской Федерации",
    "Следственный комитет РФ",
    "СК РФ",
    "СУ СК России",
    "Следком РФ",
    "Главное следственное управление Центрального аппарата СК РФ",
    "Следственное управление Следственного комитета РФ"
  ],

  "Министерство внутренних дел Российской Федерации": [
    "Министерство внутренних дел Российской Федерации",
    "Министерство внутренних дел РФ",
    "МВД РФ",
    "Отдел внутренних дел",
    "ОВД",
    "Районный отдел внутренних дел",
    "РОВД",
    "Городской отдел внутренних дел",
    "ГОВД",
    "Районное управление внутренних дел",
    "РУВД",
    "гу мвд россии",
    "ГЛАВНОЕ УПРАВЛЕНИЕ МИНИСТЕРСТВА ВНУТРЕННИХ ДЕЛ РОССИЙСКОЙ ФЕДЕРАЦИИ"
  ],
  "Федеральная служба безопасности Российской Федерации": [
    "Федеральная служба безопасности Российской Федерации",
    "Федеральная служба безопасности РФ",
    "ФСБ РФ",
    "ФСБ России"
  ]

}


def estimate_judicial_patterns_similarity_threshold():
  all_patterns = []
  for k in judicial_senders.keys():
    all_patterns += judicial_senders[k]

  max_similarity = 0
  for a in all_patterns:
    for b in all_patterns:
      if a != b:
        similarity_ratio = fuzz.ratio(a.lower(), b.lower())
        if similarity_ratio > max_similarity:
          max_similarity = similarity_ratio
          # print(a, '====', b, jaro_d)
  similarity_thresold = (100.0 + max_similarity) * 0.5

  # print(all_patterns)
  # print("similarity_thresold", similarity_thresold)
  return similarity_thresold


judicial_patterns_similarity_threshold = estimate_judicial_patterns_similarity_threshold()


def get_sender_judicial_org(sender: str) -> str or None:
  if sender is None:
    return None

  sender_l = sender.lower()
  max_similarity_ratio = judicial_patterns_similarity_threshold
  similar_p = None
  for k in judicial_senders.keys():
    patterns = judicial_senders[k]
    for p in patterns:
      p_l = p.lower()

      if sender_l.find(p_l) >= 0:
        return k
      similarity_ratio = fuzz.partial_ratio(sender_l, p_l)

      if similarity_ratio > max_similarity_ratio:
        print(similarity_ratio, sender_l, ' ||AND|| ', p_l)
        max_similarity_ratio = similarity_ratio
        similar_p = k
        # return k

  return similar_p


if __name__ == '__main__':
  print('estimated judical_patterns_similarity_threshold', judicial_patterns_similarity_threshold)
  print("sender:", get_sender_judicial_org("и Наследственный комитет Рф"))
