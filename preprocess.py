import string
import re

def remove_dup(text):

    def replace(match):
        m = match.group(0)
        try:
            if d[m[0]] == d[m[1]]:
                return m[0]
            else:
                return m[0] + m[1]
        except:
            return m[0] + m[1]
    
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    uniChars += string.ascii_letters
    unsignChars += string.ascii_letters

    d = {k: v for (k, v) in zip(uniChars, unsignChars)}
    return re.sub(fr'\S([{uniChars}])\1+\S*', replace, text)


def preprocess(text):
    text = remove_dup(text)
    return text
   
if __name__ == '__main__':
    text = 'mình cần bạn tăng nhiệt tvs thứ 1 lên 5 phần trăm ở phòng ăn tối 4 nháaaaaaâ'
    print(preprocess(text))
