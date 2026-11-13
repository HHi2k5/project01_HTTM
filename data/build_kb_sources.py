# -*- coding: utf-8 -*-
"""Tạo 3 tệp nguồn của Knowledge Base từ ghi chú đã diễn giải bằng tiếng Việt.

- kb_documents.json : ghi chú (passage) có nguồn web/book, URL, ngày truy cập, vị trí trong sách
- kb_triples.csv    : bộ ba (subject, relation, object) có doc_id
- kb_thresholds.csv : ngưỡng xét nghiệm có cấu trúc để Rule đối chiếu

Phần T/M/X kế thừa kho kiến thức Tiểu luận 03 của Hoàng Hải (data.json).
Phần N (web) và B (book) được viết mới cho Project 01 từ các trang và sách đã đọc ngày 2026-11-05.
Nội dung là diễn giải ngắn, không chép nguyên văn nguồn.
"""
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).parent
OLD = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "TIEULUAN_03/bo_sung_kg_fpt/data.json"
ACC = "2026-11-05"
U = "https://www.niddk.nih.gov/health-information/diabetes/overview/"
URL = dict(
    TESTS=U + "tests-diagnosis",
    PREV=U + "preventing-type-2-diabetes",
    HYPO=U + "preventing-problems/low-blood-glucose-hypoglycemia",
    MANAGE=U + "managing-diabetes",
    LIVING=U + "healthy-living-with-diabetes",
    PROBLEMS=U + "preventing-problems",
    WHO="https://www.who.int/news-room/fact-sheets/detail/diabetes",
    PAG="https://odphp.health.gov/sites/default/files/2019-09/Physical_Activity_Guidelines_2nd_edition.pdf",
)
BOOK = "Physical Activity Guidelines for Americans, 2nd edition (HHS, 2018)"

# (doc_id, title, source_key, text, locator)
NOTES = [
 ("N01","Ngưỡng HbA1c","TESTS","HbA1c dưới 5,7% là bình thường; từ 5,7% đến 6,4% là tiền đái tháo đường; từ 6,5% trở lên thuộc ngưỡng đái tháo đường khi không mang thai.",""),
 ("N02","Ngưỡng đường huyết lúc đói","TESTS","Đường huyết huyết tương lúc đói từ 99 mg/dL trở xuống là bình thường; 100 đến 125 mg/dL là tiền đái tháo đường; từ 126 mg/dL trở lên thuộc ngưỡng đái tháo đường. Xét nghiệm cần nhịn ăn ít nhất 8 giờ, chỉ uống vài ngụm nước.",""),
 ("N03","Ngưỡng nghiệm pháp dung nạp glucose","TESTS","Với OGTT đo sau 2 giờ, từ 139 mg/dL trở xuống là bình thường; 140 đến 199 mg/dL là tiền đái tháo đường; từ 200 mg/dL trở lên thuộc ngưỡng đái tháo đường.",""),
 ("N04","Đường huyết ngẫu nhiên","TESTS","Đường huyết ngẫu nhiên có thể đo bất kỳ lúc nào và được dùng khi đã có triệu chứng. Kết quả từ 200 mg/dL trở lên thuộc ngưỡng đái tháo đường.",""),
 ("N05","Cần xét nghiệm xác nhận","TESTS","Bác sĩ thường dùng xét nghiệm thứ hai để xác nhận đái tháo đường. Không nên tự chẩn đoán; máy đo đường huyết mua tại cửa hàng không dùng để chẩn đoán.",""),
 ("N06","Ai nên xét nghiệm","TESTS","Người có triệu chứng cần được xét nghiệm. Nên xét nghiệm định kỳ type 2 nếu từ 35 tuổi trở lên, thừa cân hoặc béo phì kèm thêm yếu tố nguy cơ, hoặc từng mắc đái tháo đường thai kỳ.",""),
 ("N07","Tần suất xét nghiệm lại","TESTS","Người có kết quả bình thường nên xét nghiệm lại sau mỗi 3 năm. Người được chẩn đoán tiền đái tháo đường nên xét nghiệm type 2 hằng năm.",""),
 ("N08","Xét nghiệm khi mang thai","TESTS","Phụ nữ mang thai chưa có chẩn đoán đái tháo đường nên được xét nghiệm đái tháo đường thai kỳ ở tuần 24 đến 28. Sau sinh nên kiểm tra type 2, thường trong vòng 12 tuần.",""),
 ("N09","Khi HbA1c có thể không chính xác","TESTS","HbA1c có thể không chính xác trong tam cá nguyệt thứ hai, thứ ba của thai kỳ, khi có một số dạng thiếu máu hoặc biến thể hemoglobin. Khi HbA1c và đường huyết không khớp, bác sĩ cần xem xét độ tin cậy của xét nghiệm.",""),
 ("N10","Giảm cân để phòng type 2","PREV","Giảm 5 đến 7 phần trăm cân nặng ban đầu và duy trì có thể giúp phòng hoặc trì hoãn type 2. Nguồn dẫn nghiên cứu Diabetes Prevention Program.",""),
 ("N11","Vận động để phòng type 2","PREV","Để giảm nguy cơ type 2, nên vận động ít nhất 30 phút mỗi ngày, 5 ngày một tuần. Người chưa quen vận động nên hỏi nhân viên y tế và tăng dần.",""),
 ("N12","Ăn uống để phòng type 2","PREV","Ăn khẩu phần nhỏ hơn, chọn thực phẩm ít chất béo và uống nước thay cho đồ uống có đường giúp giảm năng lượng nạp vào và hỗ trợ giảm cân.",""),
 ("N13","Khi có tiền đái tháo đường","PREV","Tiền đái tháo đường làm tăng khả năng tiến triển thành type 2 nhưng có thể giảm nguy cơ bằng giảm cân khi cần, vận động nhiều hơn và ăn giảm năng lượng. Người bệnh có thể hỏi bác sĩ về metformin để phòng hoặc trì hoãn type 2.",""),
 ("N14","Sau đái tháo đường thai kỳ","PREV","Đái tháo đường thai kỳ thường hết sau sinh nhưng người mẹ có nguy cơ type 2 cao hơn trong 5 đến 10 năm. Con cũng có thể dễ béo phì và mắc type 2 về sau.",""),
 ("N15","Hạ đường huyết là gì","HYPO","Hạ đường huyết là khi glucose máu giảm dưới mức an toàn, với nhiều người mắc đái tháo đường là dưới 70 mg/dL. Mức cụ thể cần hỏi đội ngũ điều trị.",""),
 ("N16","Triệu chứng hạ đường huyết","HYPO","Hạ đường huyết nhẹ đến vừa có thể gây run, đói, mệt, chóng mặt, lú lẫn, cáu gắt, tim đập nhanh, đau đầu. Mức rất thấp có thể gây mất ý thức hoặc co giật.",""),
 ("N17","Ai dễ bị hạ đường huyết","HYPO","Nguy cơ hạ đường huyết cao hơn ở người mắc type 1, người dùng insulin hoặc một số thuốc đái tháo đường, người từ 65 tuổi, người từng bị hạ đường huyết và người có bệnh thận, bệnh tim hoặc suy giảm nhận thức.",""),
 ("N18","Xử trí hạ đường huyết nhẹ","HYPO","Khi có triệu chứng hoặc đường huyết dưới 70 mg/dL: dùng ngay 15 đến 20 gam glucose hoặc carbohydrate, chờ 15 phút rồi đo lại, lặp lại nếu vẫn thấp.",""),
 ("N19","Hạ đường huyết nặng","HYPO","Hạ đường huyết nặng là khi người bệnh không tự xử trí được; có thể gây ngất, hôn mê, tử vong. Glucagon là cách điều trị tốt nhất và người xung quanh cần gọi cấp cứu ngay.",""),
 ("N20","Mục tiêu ABC","MANAGE","Quản lý ABC gồm HbA1c, huyết áp và cholesterol, cùng ngừng hút thuốc. Nhiều người có mục tiêu HbA1c dưới 7% và huyết áp dưới 130/80 mmHg; mục tiêu cá nhân do đội ngũ điều trị quyết định.",""),
 ("N21","Mục tiêu đường huyết hằng ngày","MANAGE","Mục tiêu thường dùng là 80 đến 130 mg/dL trước ăn và dưới 180 mg/dL khoảng 2 giờ sau khi bắt đầu ăn. Người lớn tuổi hoặc có bệnh khác có thể có mục tiêu khác.",""),
 ("N22","Dùng thuốc đúng chỉ định","MANAGE","Thuốc phụ thuộc loại đái tháo đường, mức kiểm soát đường huyết và bệnh kèm. Nên dùng thuốc như đã kê kể cả khi thấy khỏe, không tự giảm thuốc vì chi phí; hãy trao đổi với đội ngũ điều trị khi có tác dụng phụ.",""),
 ("N23","Thời gian trong khoảng mục tiêu","MANAGE","Với CGM, nhiều người có khoảng mục tiêu 70 đến 180 mg/dL và cố gắng ở trong khoảng này ít nhất 70% thời gian.",""),
 ("N24","Phương pháp đĩa ăn","LIVING","Phương pháp đĩa ăn dành một nửa đĩa cho rau không tinh bột, một phần tư cho thực phẩm chứa carbohydrate giàu chất xơ và một phần tư cho thực phẩm giàu đạm, giúp kiểm soát khẩu phần.",""),
 ("N25","Đồ uống và rượu","LIVING","Nên chọn đồ uống ít hoặc không có đường. Cần hạn chế rượu; với người dùng insulin hoặc một số thuốc, rượu có thể gây hạ đường huyết, nên ăn khi uống và đo đường huyết sau đó.",""),
 ("N26","Vận động khi có đái tháo đường","LIVING","Đa số người trưởng thành nên có ít nhất 150 phút hoạt động thể lực cường độ vừa mỗi tuần, khoảng 30 phút mỗi ngày trong đa số các ngày; có thể chia nhỏ.",""),
 ("N27","Giấc ngủ","LIVING","Ngủ đủ 7 đến 8 giờ mỗi đêm là một phần của lối sống lành mạnh khi quản lý đái tháo đường.",""),
 ("N28","Hút thuốc","LIVING","Đái tháo đường và hút thuốc đều làm hẹp mạch máu, cùng làm tăng nguy cơ đau tim, đột quỵ, tổn thương thần kinh, bệnh thận, bệnh mắt và cắt cụt chi. Người hút thuốc nên ngừng và tìm hỗ trợ.",""),
 ("N29","Biến chứng thận","PROBLEMS","Bệnh thận do đái tháo đường có thể được bảo vệ bằng quản lý đường huyết và đạt mục tiêu huyết áp.",""),
 ("N30","Biến chứng mắt","PROBLEMS","Đái tháo đường có thể gây giảm thị lực và mù lòa. Nên khám mắt giãn đồng tử ít nhất mỗi năm một lần.",""),
 ("N31","Biến chứng bàn chân","PROBLEMS","Tổn thương thần kinh và lưu thông máu kém có thể gây vấn đề nghiêm trọng ở bàn chân. Nên kiểm tra bàn chân ở mỗi lần khám.",""),
 ("N32","Tim mạch và đột quỵ","PROBLEMS","Đái tháo đường có thể làm tổn thương mạch máu, dẫn đến bệnh tim và đột quỵ. Quản lý đường huyết, huyết áp, cholesterol và không hút thuốc giúp phòng ngừa.",""),
 ("N33","Gánh nặng toàn cầu","WHO","WHO cho biết số người sống với đái tháo đường tăng từ 200 triệu năm 1990 lên 830 triệu năm 2022; 14% người từ 18 tuổi sống với bệnh năm 2022.",""),
 ("N34","Type 2 chiếm đa số","WHO","Theo WHO, hơn 95% người mắc đái tháo đường là type 2. Triệu chứng type 2 có thể nhẹ và nhiều năm mới được nhận ra, nên phát hiện sớm dựa vào khám và xét nghiệm định kỳ.",""),
 ("N35","Phòng ngừa theo WHO","WHO","Ăn uống lành mạnh, vận động đều, giữ cân nặng bình thường và tránh thuốc lá giúp phòng hoặc trì hoãn type 2.",""),
 ("B01","Khuyến nghị vận động cho người trưởng thành","PAG","Người trưởng thành nên có 150 đến 300 phút hoạt động cường độ vừa, hoặc 75 đến 150 phút cường độ mạnh mỗi tuần, cộng thêm hoạt động tăng cơ ít nhất 2 ngày mỗi tuần.","Sách, trang PDF 8"),
 ("B02","Vận động và nguy cơ type 2","PAG","Vận động đều làm giảm mạnh nguy cơ type 2 ở mọi vóc dáng; lợi ích bắt đầu cả khi chưa đạt 150 phút mỗi tuần và tăng thêm khi vận động nhiều hơn. Chỉ một buổi vận động đã có thể cải thiện độ nhạy insulin.","Sách, trang PDF 36"),
 ("B03","Người chưa vận động đủ","PAG","Người chưa đạt 150 phút mỗi tuần nên tăng dần đến mục tiêu.","Sách, trang PDF 62"),
 ("B04","Người cao tuổi có bệnh mạn tính","PAG","Người cao tuổi không đạt được 150 phút mỗi tuần vì bệnh mạn tính nên vận động trong khả năng và tình trạng cho phép.","Sách, trang PDF 9 và 68"),
 ("B05","Vận động khi mang thai","PAG","Phụ nữ mang thai và sau sinh nên có ít nhất 150 phút hoạt động cường độ vừa mỗi tuần.","Sách, trang PDF 9 và 80"),
 ("B06","Vận động cho người mắc type 2","PAG","Với người mắc type 2, vận động có tác dụng điều trị, giảm bệnh kèm và giúp phòng các yếu tố làm bệnh tiến triển, nên thường được khuyến nghị.","Sách, trang PDF 83"),
]

# (subject, relation, object, doc_id)
NEW_TRIPLES = [
 ("Đái tháo đường","chẩn_đoán_bằng","HbA1c","N01"),("Tiền đái tháo đường","chẩn_đoán_bằng","HbA1c","N01"),
 ("Đái tháo đường","chẩn_đoán_bằng","Đường huyết đói","N02"),("Tiền đái tháo đường","chẩn_đoán_bằng","Đường huyết đói","N02"),
 ("Đái tháo đường","chẩn_đoán_bằng","OGTT","N03"),("Tiền đái tháo đường","chẩn_đoán_bằng","OGTT","N03"),
 ("Đái tháo đường","chẩn_đoán_bằng","Đường huyết ngẫu nhiên","N04"),
 ("Chẩn đoán đái tháo đường","cần","Xét nghiệm xác nhận","N05"),
 ("Tuổi từ 35","là_chỉ_định_xét_nghiệm","Type 2","N06"),("Thừa cân hoặc béo phì","là_chỉ_định_xét_nghiệm","Type 2","N06"),
 ("Tiền sử đái tháo đường thai kỳ","là_chỉ_định_xét_nghiệm","Type 2","N06"),
 ("Tiền đái tháo đường","cần_theo_dõi","Xét nghiệm hằng năm","N07"),("Xét nghiệm bình thường","cần_theo_dõi","Xét nghiệm lại sau 3 năm","N07"),
 ("Đái tháo đường thai kỳ","thường_được_chẩn_đoán_ở","Tuần thai 24–28","N08"),
 ("Thiếu máu","làm_sai_lệch","HbA1c","N09"),("Biến thể hemoglobin","làm_sai_lệch","HbA1c","N09"),("Thai kỳ","làm_sai_lệch","HbA1c","N09"),
 ("Giảm 5–7% cân nặng","giúp_phòng_hoặc_trì_hoãn","Type 2","N10"),
 ("Hoạt động thể lực","giúp_phòng_hoặc_trì_hoãn","Type 2","N11"),
 ("Giảm đồ uống có đường","giúp_phòng_hoặc_trì_hoãn","Type 2","N12"),("Giảm khẩu phần","giúp_phòng_hoặc_trì_hoãn","Type 2","N12"),
 ("Tiền đái tháo đường","có_thể_tiến_triển","Type 2","N13"),
 ("Đái tháo đường thai kỳ","tăng_nguy_cơ_về_sau","Type 2","N14"),
 ("Đái tháo đường","có_thể_gây","Hạ đường huyết","N15"),
 ("Hạ đường huyết","có_thể_có_triệu_chứng","Run","N16"),("Hạ đường huyết","có_thể_có_triệu_chứng","Chóng mặt","N16"),
 ("Hạ đường huyết","có_thể_có_triệu_chứng","Lú lẫn","N16"),("Hạ đường huyết","có_thể_có_triệu_chứng","Tim đập nhanh","N16"),
 ("Hạ đường huyết","có_thể_có_triệu_chứng","Đau đầu","N16"),("Hạ đường huyết","có_thể_có_triệu_chứng","Mất ý thức","N16"),
 ("Hạ đường huyết","có_thể_có_triệu_chứng","Co giật","N16"),
 ("Insulin","tăng_nguy_cơ","Hạ đường huyết","N17"),("Tuổi từ 65","tăng_nguy_cơ","Hạ đường huyết","N17"),
 ("Bệnh thận do diabetes","tăng_nguy_cơ","Hạ đường huyết","N17"),
 ("Hạ đường huyết","xử_trí_bằng","Dùng glucose nhanh và đo lại","N18"),
 ("Hạ đường huyết","xử_trí_bằng","Glucagon","N19"),("Hạ đường huyết","có_thể_dẫn_đến","Cấp cứu y tế","N19"),
 ("Quản lý Type 2","bao_gồm_kiểm_soát","HbA1c","N20"),("Quản lý Type 2","bao_gồm_kiểm_soát","Ngừng hút thuốc","N20"),
 ("Đái tháo đường","theo_dõi_bằng","CGM","N23"),("Đái tháo đường","theo_dõi_bằng","Máy đo đường huyết tại nhà","N21"),
 ("Quản lý Type 2","sử_dụng","Phương pháp đĩa ăn","N24"),("Quản lý Type 2","sử_dụng","Giấc ngủ 7–8 giờ","N27"),
 ("Rượu","có_thể_gây","Hạ đường huyết","N25"),
 ("Hút thuốc","tăng_nguy_cơ","Bệnh tim","N28"),("Hút thuốc","tăng_nguy_cơ","Đột quỵ","N28"),("Hút thuốc","tăng_nguy_cơ","Bệnh thận do diabetes","N28"),
 ("Hút thuốc","tăng_nguy_cơ","Bệnh mắt do diabetes","N28"),("Hút thuốc","tăng_nguy_cơ","Tổn thương thần kinh","N28"),
 ("Đái tháo đường","có_thể_gây","Bệnh thận do diabetes","N29"),("Đái tháo đường","có_thể_gây","Bệnh mắt do diabetes","N30"),
 ("Bệnh mắt do diabetes","cần_theo_dõi","Khám mắt hằng năm","N30"),("Vấn đề bàn chân","cần_theo_dõi","Khám bàn chân mỗi lần tái khám","N31"),
 ("Huyết áp cao","tăng_nguy_cơ","Bệnh thận do diabetes","N29"),
 ("Đái tháo đường","có_thể_gây","Vấn đề bàn chân","N31"),("Đái tháo đường","có_thể_gây","Bệnh tim","N32"),("Đái tháo đường","có_thể_gây","Đột quỵ","N32"),
 ("Giữ cân nặng bình thường","giúp_phòng_hoặc_trì_hoãn","Type 2","N35"),("Tránh thuốc lá","giúp_phòng_hoặc_trì_hoãn","Type 2","N35"),
 ("Hoạt động tăng cơ","hỗ_trợ","Quản lý Type 2","B01"),("Hoạt động thể lực","cải_thiện","Nhạy cảm insulin","B02"),
 ("Hoạt động thể lực","hỗ_trợ","Quản lý Type 2","B06"),
 # Cầu nối DB - KB: cột dữ liệu sang thực thể
 ("age","ánh_xạ_tới","Tuổi từ 35","N06"),("bmi","ánh_xạ_tới","Thừa cân hoặc béo phì","N06"),
 ("HbA1c_level","ánh_xạ_tới","HbA1c","N01"),("blood_glucose_level","ánh_xạ_tới","Đường huyết đói","N02"),
 ("blood_glucose_level","ánh_xạ_tới","Đường huyết ngẫu nhiên","N04"),("smoking_history","ánh_xạ_tới","Hút thuốc","N28"),
 ("hypertension","ánh_xạ_tới","Huyết áp cao","N29"),("heart_disease","ánh_xạ_tới","Bệnh tim","N32"),
]

# test, category, lower (>=), upper (<), unit, doc_id
THRESHOLDS = [
 ("HbA1c","Bình thường",None,5.7,"%","N01"),("HbA1c","Tiền đái tháo đường",5.7,6.5,"%","N01"),("HbA1c","Ngưỡng đái tháo đường",6.5,None,"%","N01"),
 ("Đường huyết đói","Bình thường",None,100,"mg/dL","N02"),("Đường huyết đói","Tiền đái tháo đường",100,126,"mg/dL","N02"),("Đường huyết đói","Ngưỡng đái tháo đường",126,None,"mg/dL","N02"),
 ("OGTT 2 giờ","Bình thường",None,140,"mg/dL","N03"),("OGTT 2 giờ","Tiền đái tháo đường",140,200,"mg/dL","N03"),("OGTT 2 giờ","Ngưỡng đái tháo đường",200,None,"mg/dL","N03"),
 ("Đường huyết ngẫu nhiên","Ngưỡng đái tháo đường (khi có triệu chứng)",200,None,"mg/dL","N04"),
 ("Đường huyết","Hạ đường huyết",None,70,"mg/dL","N15"),
]

TYPES = {
 "Disease":"Đái tháo đường|Type 1|Type 2|Đái tháo đường thai kỳ|Đái tháo đường đơn gen|MODY|Đái tháo đường sơ sinh|Đái tháo đường sơ sinh thoáng qua|Đái tháo đường sơ sinh vĩnh viễn|Tiền đái tháo đường|Hạ đường huyết",
 "Complication":"Bệnh mắt do diabetes|Bệnh nướu|Bệnh thận do diabetes|Bệnh tim|Đột quỵ|Vấn đề bàn chân|Tổn thương thần kinh|Tổn thương mạch máu|Nhiễm toan ceton|Tiền sản giật|Sinh non|Cân nặng khi sinh lớn|Hạ đường huyết sau sinh|Khó thở sau sinh|Cấp cứu y tế",
 "Symptom":"Khát nhiều|Tiểu nhiều|Mệt mỏi|Nhìn mờ|Đói nhiều|Vết thương lâu lành|Sụt cân không rõ nguyên nhân|Nhiễm trùng thường xuyên|Run|Chóng mặt|Lú lẫn|Tim đập nhanh|Đau đầu|Mất ý thức|Co giật|Không có triệu chứng",
 "Test":"HbA1c|OGTT|Đường huyết đói|Đường huyết ngẫu nhiên|CGM|Máy đo đường huyết tại nhà|Xét nghiệm di truyền|Xét nghiệm xác nhận|Xét nghiệm hằng năm|Xét nghiệm lại sau 3 năm|Xét nghiệm bình thường|Khám mắt hằng năm|Khám bàn chân mỗi lần tái khám",
 "Drug":"Metformin|Insulin|Acarbose|Pioglitazone|Glucagon|Insulin tác dụng dài|Insulin tác dụng ngắn|Insulin tác dụng nhanh|Insulin tác dụng siêu dài|Insulin tác dụng trung bình",
 "DrugClass":"Biguanide|Sulfonylurea|TZD|Glinide|Ức chế DPP-4|Ức chế SGLT2|Ức chế alpha-glucosidase|Đồng vận thụ thể GLP-1",
 "RiskFactor":"Tuổi tăng|Tuổi từ 35|Tuổi từ 65|Thừa cân hoặc béo phì|Tiền sử gia đình có diabetes|Ít hoạt động thể lực|Vòng eo lớn|Yếu tố di truyền|Mẹ bị diabetes thai kỳ khi mang thai|Tiền sử đái tháo đường thai kỳ|Hút thuốc|Huyết áp cao|Rượu|Thiếu máu|Biến thể hemoglobin",
 "Lifestyle":"Hoạt động thể lực|Quản lý cân nặng|Giảm 5–7% cân nặng|Giảm đồ uống có đường|Giảm khẩu phần|Ngừng hút thuốc|Phương pháp đĩa ăn|Giấc ngủ 7–8 giờ|Giữ cân nặng bình thường|Tránh thuốc lá|Hoạt động tăng cơ|Dùng glucose nhanh và đo lại",
 "Feature":"age|bmi|HbA1c_level|blood_glucose_level|smoking_history|hypertension|heart_disease",
}

def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    docs = []
    for d in old["medical_docs"]:
        docs.append(dict(doc_id=d["doc_id"], title=d["title"], source_type="web", publisher=d["source"], url=d["url"],
                         accessed=d["accessed"], locator="", text=d["text"], origin="Tiểu luận 03"))
    for doc_id, title, key, text, loc in NOTES:
        pub = BOOK if key == "PAG" else ("WHO" if key == "WHO" else "NIDDK")
        docs.append(dict(doc_id=doc_id, title=title, source_type="book" if key == "PAG" else "web", publisher=pub,
                         url=URL[key], accessed=ACC, locator=loc, text=text, origin="Project 01"))
    (HERE / "kb_documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")

    etype = {e: t for t, names in TYPES.items() for e in names.split("|")}
    seen, rows = set(), []
    for f in old["medical_facts"]:
        rows.append((f["subject"], f["relation"], f["object"], f["doc_id"]))
    rows += NEW_TRIPLES
    ids = {d["doc_id"] for d in docs}
    with open(HERE / "kb_triples.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["subject", "subject_type", "relation", "object", "object_type", "doc_id"])
        for s, r, o, d in rows:
            assert d in ids, d
            if (s, r, o) in seen: continue
            seen.add((s, r, o))
            w.writerow([s, etype.get(s, "Concept"), r, o, etype.get(o, "Concept"), d])
    with open(HERE / "kb_thresholds.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["test", "category", "lower", "upper", "unit", "doc_id"]); w.writerows(THRESHOLDS)
    print(len(docs), "notes;", len(seen), "triples;", len(THRESHOLDS), "thresholds")

if __name__ == "__main__":
    main()
