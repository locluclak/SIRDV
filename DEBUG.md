Chạy examples\demo_si.py
Seed đã được cố định nên kết quả sẽ ra giống nhau giữa các lần chạy
Source code nằm trong si_for_cg/

Phương pháp dùng cho data valuation là LAVA trong si_for_cg/lava_function

Flow code:
demo_si.py -> si_dv.py -> si_runner.py -> si_overconditioning.py/si_parametric.py

parametric hiện không được dùng vì khó debug + chậm. Thay vào đó dùng hàm overconditioning lên z_obs và oc lên z bất kì
```python
    print('oc original', intervals)
    print()
    print('oc with specific z', ocwithz_intervals(Xtrain, Xval, decomposition, 
                                    adapter, observed_selection, config,z=0.28082)) #<-- z <= 0.28082 lỗi
                                                                                    #    z >= 0.28083 không lỗi
```

Vấn đề: Khi giải OC z_obs trên OT thì có được interval $A = [z1, z2]$, nhưng giải $z != zobs, z \in A$ thì lại được interval B $B \neq A$ như phía dưới 

Do OT sai nên khéo theo các điều kiện khác cũng sai.

Nguyên nhân là do Linear Programming (trong OptimalTransport.py) ở z_obs và z giải ra 2 tập Basis khác nhau. Còn tại sao lại vậy thì chưa tìm ra.

```
(oc trên z_obs)
basis B [  9  11  27  30  32  46  50  54  56  64  74  79  88  97 109 112 115 121
 122 127 130 143 154 163 176 185 190 198 203 204 211 227 232 248 251 268
 275 286 295]
OC OT itv [(np.float64(0.0897531203030525), np.float64(0.7702502613631955))]
OC selec itv [(np.float64(0.1983316756033726), np.float64(0.3919662183388167))]
oc original [(np.float64(0.1983316756033726), np.float64(0.3919662183388167))]


(oc trên z)
optimal cost1 [[3.54293722]]
optimal cost2 [[3.54293722]]
basis B [  9  11  27  30  32  46  50  54  56  64  74  79  86  88  97 109 112 115
 122 127 130 143 154 163 176 185 190 203 204 211 217 227 232 248 251 268
 275 286 295]
OC OT itv [(np.float64(-0.21797975270380168), np.float64(0.7702502613631955))]
OC selec itv [(np.float64(0.06360502979893877), np.float64(0.1506966221057207))]
oc with specific z [(np.float64(0.06360502979893877), np.float64(0.1506966221057207))]
```

Note: 
- Ban đầu bug chỉ là basis z và zobs khác nhau, nhưng interval giống nhau. Lúc đó lí giải là bài toán LP có thể giải ra nhiều tập basis set khác nhau nhưng vẫn cho optimal cost giống nhau nên interval OC vẫn giống nhau do thỏa dk tối ưu.
- Nhưng bug ở trên lại bác bỏ suy đoán lúc đầu.