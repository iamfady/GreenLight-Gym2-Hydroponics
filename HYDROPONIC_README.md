# 🌱 Hydroponic Lettuce Environment for GreenLight Gym

## 📋 نظرة عامة

تم تطوير بيئة الهيدروبونيك للخس كتطوير متقدم لبيئة GreenLight Gym الأصلية. هذه البيئة مصممة خصيصاً لزراعة الخس في أنظمة الهيدروبونيك مع إدارة متقدمة للمغذيات والتحكم في درجة الحموضة والتوصيلية الكهربائية.

## 🚀 الميزات الجديدة

### 1. **إدارة المغذيات المتقدمة**
- **المغذيات الأساسية**: N, P, K, Ca, Mg, S
- **المغذيات الدقيقة**: Fe, Mn, Zn, Cu, B, Mo
- **مراقبة مستويات المغذيات** في الوقت الفعلي
- **كشف نقص المغذيات** تلقائياً

### 2. **التحكم في درجة الحموضة (pH)**
- **النطاق الأمثل**: 5.5 - 6.5
- **التحكم التلقائي** في درجة الحموضة
- **عقوبات** عند الخروج عن النطاق الأمثل

### 3. **التحكم في التوصيلية الكهربائية (EC)**
- **النطاق الأمثل**: 1.2 - 2.2 mS/cm
- **مراقبة مستمرة** للتوصيلية
- **ضبط تلقائي** لمستويات المغذيات

### 4. **إدارة درجة حرارة المحلول**
- **النطاق الأمثل**: 18°C - 24°C
- **مراقبة درجة الحرارة** مع التهوية
- **تحكم في التدفئة والتبريد**

### 5. **إدارة الأكسجين**
- **مستوى الأكسجين الأمثل**: 6-12 mg/L
- **تحكم في معدل التهوية**
- **مراقبة استهلاك الأكسجين**

## 🏗️ الأنظمة المدعومة

### 1. **NFT (Nutrient Film Technique)**
- تقنية الفيلم المغذي
- تدفق مستمر للمحلول
- كفاءة عالية في استخدام المياه

### 2. **DWC (Deep Water Culture)**
- زراعة المياه العميقة
- تهوية قوية
- نمو سريع للجذور

### 3. **Aeroponics**
- زراعة الهواء
- رش المغذيات
- نمو أسرع

## 📊 معاملات البيئة

### **المعاملات الأساسية**
```yaml
nx: 30          # عدد حالات النظام (28 + 2 للـ pH و EC)
nu: 8           # عدد مدخلات التحكم (6 + 2 للـ pH و EC)
nd: 10          # عدد الاضطرابات المناخية
dt: 900         # خطوة زمنية (ثانية)
```

### **قيود النظام**
```yaml
# قيود المناخ
temp_min: 15°C, temp_max: 28°C
rh_min: 50%, rh_max: 85%
co2_min: 300 ppm, co2_max: 1300 ppm

# قيود الهيدروبونيك
ph_min: 5.5, ph_max: 6.5
ec_min: 1.2 mS/cm, ec_max: 2.2 mS/cm
solution_temp_min: 18°C, solution_temp_max: 24°C
```

## 🎯 كيفية الاستخدام

### 1. **إنشاء البيئة الأساسية**
```python
from gl_gym.environments.Hydroponic_Lettuce_Env import HydroponicLettuceEnv

env = HydroponicLettuceEnv(
    hydroponic_params={
        "system_type": "NFT",
        "target_ph": 6.0,
        "target_ec": 1.4
    }
)
```

### 2. **إنشاء البيئة مع معاملات مخصصة**
```python
env = HydroponicLettuceEnv(
    constraints={
        "ph_min": 5.8,
        "ph_max": 6.2,
        "ec_min": 1.5,
        "ec_max": 1.8
    },
    hydroponic_params={
        "system_type": "DWC",
        "target_ph": 6.0,
        "target_ec": 1.6,
        "nutrient_mix": {
            "N": 200, "P": 60, "K": 250
        }
    }
)
```

### 3. **التحكم في النظام**
```python
# ضبط درجة الحموضة
env.adjust_ph(6.2)

# ضبط التوصيلية الكهربائية
env.adjust_ec(1.6)

# إضافة مغذيات
env.add_nutrients("N", 50)
env.add_nutrients("K", 30)

# ضبط معدل التدفق
env.set_flow_rate(3.0)

# ضبط معدل التهوية
env.set_aeration_rate(1.5)
```

### 4. **مراقبة النظام**
```python
# الحصول على معلومات الهيدروبونيك
info = env._get_hydroponic_info()

# حساب درجة صحة النظام
health_score = env.get_hydroponic_health_score()

# الحصول على التوصيات
recommendations = env.get_system_recommendations()

# الحصول على حالة النظام
status = env.get_hydroponic_status()
```

## 🔬 الميزات المتقدمة

### 1. **كشف نقص المغذيات**
```python
deficiencies = env._check_nutrient_deficiency()
for nutrient, is_deficient in deficiencies.items():
    if is_deficient:
        print(f"نقص في {nutrient}")
```

### 2. **مراقبة الانتهاكات**
```python
ph_violation = env._check_ph_violation()
ec_violation = env._check_ec_violation()
temp_violation = env._check_solution_temp_violation()
```

### 3. **تحديث المعاملات**
```python
env.update_hydroponic_params({
    "target_ph": 6.2,
    "target_ec": 1.6
})
```

## 📈 محاكاة النظام

### 1. **خطوة واحدة**
```python
obs, reward, terminated, truncated, info = env.step(action)
```

### 2. **محاكاة كاملة**
```python
obs = env.reset()
for step in range(1000):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        break
```

### 3. **مراقبة المعلومات**
```python
print(f"Reward: {reward}")
print(f"pH: {info['current_ph']}")
print(f"EC: {info['current_ec']}")
print(f"Solution Temp: {info['solution_temperature']}")
print(f"Oxygen: {info['oxygen_level']}")
```

## 🧪 اختبار النظام

### **تشغيل الاختبارات**
```bash
python test_hydroponic.py
```

### **اختبار الميزات الأساسية**
- إنشاء البيئة
- ضبط المعاملات
- مراقبة الحالة
- التحكم في النظام

### **اختبار الميزات المتقدمة**
- كشف نقص المغذيات
- مراقبة الانتهاكات
- حساب درجة الصحة
- التوصيات

## 📁 هيكل الملفات

```
gl_gym/
├── environments/
│   ├── Hydroponic_Lettuce_Env.py    # البيئة المتقدمة للهيدروبونيك
│   ├── lettuce_env.py               # البيئة الأساسية للخس
│   ├── base_env.py                  # البيئة الأساسية
│   └── parameters.py                # معاملات النظام
├── configs/
│   └── envs/
│       ├── HydroponicLettuceEnv.yml # تكوين الهيدروبونيك
│       └── LettuceEnv.yml           # تكوين الخس الأساسي
└── RL/
    └── utils.py                     # أدوات التعلم المعزز
```

## 🔧 التكوين

### **ملف التكوين الأساسي**
```yaml
HydroponicLettuceEnv:
  base: GreenLightEnv
  reward_function: GreenhouseReward
  
  observation_modules: 
    - IndoorClimateObservations
    - BasicCropObservations
    - ControlObservations
    - WeatherObservations
    - TimeObservations
    - WeatherForecastObservations
  
  constraints:
    co2_min: 300.0
    co2_max: 1300.0
    temp_min: 15.0
    temp_max: 28.0
    rh_min: 50.0
    rh_max: 85.0
    ph_min: 5.5
    ph_max: 6.5
    ec_min: 1.2
    ec_max: 2.2
    solution_temp_min: 18.0
    solution_temp_max: 24.0
```

## 🎯 أفضل الممارسات

### 1. **إدارة درجة الحموضة**
- **الخس**: 5.5 - 6.5
- **الطماطم**: 5.5 - 6.8
- **الخيار**: 5.5 - 6.0

### 2. **إدارة التوصيلية الكهربائية**
- **الشتلات**: 0.8 - 1.2 mS/cm
- **النمو**: 1.2 - 1.8 mS/cm
- **الإنتاج**: 1.8 - 2.2 mS/cm

### 3. **إدارة درجة الحرارة**
- **المحلول**: 18°C - 24°C
- **الهواء**: 15°C - 28°C
- **الرطوبة**: 50% - 85%

### 4. **إدارة المغذيات**
- **النيتروجين**: 150-200 ppm
- **الفوسفور**: 50-80 ppm
- **البوتاسيوم**: 200-300 ppm

## 🚨 استكشاف الأخطاء

### **مشاكل شائعة**

1. **انخفاض درجة الحموضة**
   - إضافة pH up
   - فحص جودة المياه
   - مراجعة المغذيات

2. **ارتفاع التوصيلية الكهربائية**
   - تخفيف المحلول
   - فحص تركيز المغذيات
   - مراجعة معدل التدفق

3. **انخفاض مستوى الأكسجين**
   - زيادة معدل التهوية
   - فحص مضخة الهواء
   - تنظيف فتحات التهوية

4. **نقص المغذيات**
   - إضافة المغذيات المطلوبة
   - فحص معدل التدفق
   - مراجعة جدول التغذية

## 📚 المراجع

- [Hydroponic Lettuce Production Guide](https://extension.psu.edu/hydroponic-lettuce-production)
- [Nutrient Management in Hydroponics](https://www.hortidaily.com/article/6000000/nutrient-management-in-hydroponics/)
- [pH and EC Management](https://www.cropking.com/blog/ph-ec-management-hydroponics)

## 🤝 المساهمة

نرحب بالمساهمات! يرجى:
1. Fork المشروع
2. إنشاء فرع للميزة الجديدة
3. إضافة الاختبارات
4. إرسال Pull Request

## 📄 الترخيص

هذا المشروع مرخص تحت رخصة MIT. راجع ملف `LICENSE` للتفاصيل.

---

**🌱 تم تطوير هذا النظام بواسطة فريق GreenLight Gym لدعم زراعة الهيدروبونيك المستدامة**


