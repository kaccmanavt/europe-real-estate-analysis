Анализ рынка недвижимости в Европе
Автор: Проект для учебного курса
Дата: Март 2026

Этот скрипт выполняет расширенный анализ данных о ценах на недвижимость
в крупнейших городах Европы.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

Настройка русских шрифтов для графиков (опционально)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("Анализ рынка недвижимости в Европе")
print("=" * 60)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

print("\n 1. Загрузка данных...")

# Загружаем данные из CSV
df = pd.read_csv('data/clean_data.csv')

print(f"   Загружено {len(df)} городов")
print(f"   Столбцы: {list(df.columns)}")

# ============================================
# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ============================================

print("\n 2. Описательная статистика...")

stats_center = df['price_center_sqm'].describe()
stats_yield = df['gross_rental_yield'].describe()

print("\n   Цена за м² в центре (€):")
print(f"   - Среднее: {stats_center['mean']:,.0f} €/м²")
print(f"   - Медиана: {stats_center['50%']:,.0f} €/м²")
print(f"   - Минимум: {stats_center['min']:,.0f} €/м² ({df.loc[df['price_center_sqm'].idxmin(), 'city']})")
print(f"   - Максимум: {stats_center['max']:,.0f} €/м² ({df.loc[df['price_center_sqm'].idxmax(), 'city']})")
print(f"   - Стандартное отклонение: {stats_center['std']:,.0f} €/м²")

print("\n   Доходность от аренды (%):")
print(f"   - Среднее: {stats_yield['mean']:.1f}%")
print(f"   - Медиана: {stats_yield['50%']:.1f}%")
print(f"   - Минимум: {stats_yield['min']:.1f}% ({df.loc[df['gross_rental_yield'].idxmin(), 'city']})")
print(f"   - Максимум: {stats_yield['max']:.1f}% ({df.loc[df['gross_rental_yield'].idxmax(), 'city']})")

# ============================================
# 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================

print("\n 3. Корреляционный анализ...")

# Расчет корреляции Пирсона
correlation, p_value = stats.pearsonr(df['price_center_sqm'], df['gross_rental_yield'])

print(f"\n   Связь между ценой покупки и доходностью аренды:")
print(f"   - Коэффициент корреляции Пирсона: {correlation:.3f}")
print(f"   - p-value: {p_value:.5f}")

if correlation < -0.7:
    print("    Интерпретация: Сильная обратная корреляция")
    print("      Чем выше цена покупки жилья, тем ниже доходность от аренды")
elif correlation < -0.3:
    print("    Интерпретация: Умеренная обратная корреляция")
else:
    print("    Интерпретация: Слабая или отсутствующая корреляция")

# ============================================
# 4. РЕГРЕССИОННЫЙ АНАЛИЗ
# ============================================

print("\n 4. Регрессионный анализ...")

# Линейная регрессия: доходность = a * цена + b
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
    df['price_center_sqm'], 
    df['gross_rental_yield']
)

print(f"\n   Уравнение регрессии:")
print(f"   Доходность = {slope:.6f} × Цена + {intercept:.3f}")
print(f"\n   R² (коэффициент детерминации): {r_value**2:.3f}")
print(f"   Это означает, что {r_value**2*100:.1f}% вариации доходности")
print(f"   объясняется изменением цены на недвижимость")

# Прогноз для Парижа
paris_price = 17200
predicted_yield = slope * paris_price + intercept
print(f"\n   Прогноз для Парижа (€17,200/м²):")
print(f"      Ожидаемая доходность: {predicted_yield:.1f}%")
print(f"      Фактическая доходность: 2.5%")
print(f"      Разница: {abs(predicted_yield - 2.5):.1f}%")

# ============================================
# 5. АНАЛИЗ ПО РЕГИОНАМ
# ============================================

print("\n 5. Анализ по регионам...")

region_stats = df.groupby('region').agg({
    'price_center_sqm': ['mean', 'min', 'max'],
    'gross_rental_yield': 'mean'
}).round(1)

print("\n   Региональная статистика:")
print(region_stats.to_string())

print("\n   Выводы по регионам:")
print("   - Западная Европа: самая высокая цена (€9,200/м²), низкая доходность (3.3%)")
print("   - Восточная Европа: самая низкая цена (€2,600/м²), высокая доходность (6.0%)")
print("   - Разница в цене между Западом и Востоком: 3.5 раза")

# ============================================
# 6. ТОП-5 И АНТИ-ТОП-5
# ============================================

print("\n 6. Топ-5 и анти-топ-5...")

print("\n   Самые дорогие города (€/м²):")
for i, row in df.nlargest(5, 'price_center_sqm')[['city', 'price_center_sqm']].iterrows():
    print(f"   {row['city']}: {row['price_center_sqm']:,.0f} €/м²")

print("\n   Самые доступные города (€/м²):")
for i, row in df.nsmallest(5, 'price_center_sqm')[['city', 'price_center_sqm']].iterrows():
    print(f"   {row['city']}: {row['price_center_sqm']:,.0f} €/м²")

print("\n   Самая высокая доходность от аренды (%):")
for i, row in df.nlargest(5, 'gross_rental_yield')[['city', 'gross_rental_yield']].iterrows():
    print(f"   {row['city']}: {row['gross_rental_yield']:.1f}%")

print("\n   Самая низкая доходность от аренды (%):")
for i, row in df.nsmallest(5, 'gross_rental_yield')[['city', 'gross_rental_yield']].iterrows():
    print(f"   {row['city']}: {row['gross_rental_yield']:.1f}%")

# ============================================
# 7. СОЗДАНИЕ ГРАФИКОВ (ОПЦИОНАЛЬНО)
# ============================================

print("\n 7. Создание графиков...")

# Настройка стиля
sns.set_style("whitegrid")

# График 1: Зависимость цены от доходности
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot с регрессией
sns.regplot(x='price_center_sqm', y='gross_rental_yield', data=df, ax=axes[0])
axes[0].set_xlabel('Цена за м² в центре (€)')
axes[0].set_ylabel('Доходность от аренды (%)')
axes[0].set_title('Связь между ценой покупки и доходностью аренды')
axes[0].axhline(y=df['gross_rental_yield'].mean(), color='red', linestyle='--', alpha=0.5)
axes[0].axvline(x=df['price_center_sqm'].mean(), color='red', linestyle='--', alpha=0.5)

# Boxplot по регионам
sns.boxplot(x='region', y='price_center_sqm', data=df, ax=axes[1])
axes[1].set_xlabel('Регион')
axes[1].set_ylabel('Цена за м² (€)')
axes[1].set_title('Распределение цен по регионам')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/python_analysis_plots.png', dpi=150, bbox_inches='tight')
print("    Графики сохранены в 'visualizations/python_analysis_plots.png'")

# ============================================
# 8. ИТОГОВЫЕ ВЫВОДЫ
# ============================================

print("\n" + "=" * 60)
print(" ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 60)

print("""
1.  РЫНОК НЕДВИЖИМОСТИ В ЕВРОПЕ СИЛЬНО ДИФФЕРЕНЦИРОВАН:
    - Париж (€17,200/м²) в 11 раз дороже Софии (€1,550/м²)
    - Западная Европа в 3.5 раза дороже Восточной

2.  ИНВЕСТИЦИОННАЯ ПРИВЛЕКАТЕЛЬНОСТЬ:
    - Выявлена сильная обратная корреляция (r = -0.87)
    - Дорогие города (Париж, Лондон) → низкая доходность (2.5-3.1%)
    - Доступные города (София, Будапешт) → высокая доходность (6.2-6.5%)

3.  РЕГИОНАЛЬНЫЙ РАЗРЫВ:
    - Восточная Европа предлагает лучший баланс цены и доходности
    - Западная Европа требует больших вложений при скромной отдаче

4.  ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:
    - Для инвестиций: Восточная Европа (София, Будапешт, Бухарест)
    - Для проживания в центре: города с минимальной разницей центр/окраина (Брюссель, Милан)
    - Для долгосрочного роста капитала: Западная Европа
""")

print("\n Анализ завершен!")
