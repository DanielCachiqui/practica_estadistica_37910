import os
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
from scipy import stats
import tkinter as tk
from tkinter import ttk

plt.rcParams.update({
    'font.family'       : 'Arial',
    'axes.titlesize'    : 13,
    'axes.titleweight'  : 'bold',
    'axes.titlecolor'   : '#1F3864',
    'axes.labelsize'    : 10,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'figure.facecolor'  : '#F4F6FA',
    'axes.facecolor'    : '#FFFFFF',
})

COLORES = {
    'barras' : ['#2E75B6', '#70AD47', '#ED7D31', '#C00000', '#7030A0'],
    'torta'  : ['#2E75B6', '#70AD47', '#ED7D31'],
    'linea'  : '#ED7D31',
    'hist'   : '#2E75B6',
    'ojiva'  : '#7030A0',
    'rojo'   : '#C00000',
    'verde'  : '#70AD47',
    'morado' : '#7030A0',
}

COLOR_SECCION = {
    'central'   : '#DEEAF1',
    'posicion'  : '#E2EFDA',
    'dispersion': '#FCE4D6',
    'forma'     : '#EDE7F6',
}


def cargar_datos(ruta='datos_estudiantes.csv'):
    df = pd.read_csv(ruta, encoding='latin1')
    df.columns = ['id', 'carrera', 'genero',
                  'materias_aprobadas', 'edad', 'promedio_acumulado']
    return df


def tabla_cualitativa(df, columna):
    n   = len(df)
    fi  = df[columna].value_counts().sort_index()
    hi  = (fi / n).round(4)
    pct = (hi * 100).round(2)
    Fi  = fi.cumsum()
    Hi  = hi.cumsum().round(4)
    tabla = pd.DataFrame({
        columna.capitalize(): fi.index,
        'fi': fi.values, 'hi': hi.values,
        '%' : pct.values, 'Fi': Fi.values, 'Hi': Hi.values,
    })
    total = pd.DataFrame([{
        columna.capitalize(): 'TOTAL',
        'fi': fi.sum(), 'hi': round(hi.sum(), 4),
        '%' : round(pct.sum(), 2), 'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True)


def tabla_discreta(df, columna):
    n   = len(df)
    fi  = df[columna].value_counts().sort_index()
    hi  = (fi / n).round(4)
    pct = (hi * 100).round(2)
    Fi  = fi.cumsum()
    Hi  = hi.cumsum().round(4)
    tabla = pd.DataFrame({
        columna: fi.index,
        'fi': fi.values, 'hi': hi.values,
        '%' : pct.values, 'Fi': Fi.values, 'Hi': Hi.values,
    })
    total = pd.DataFrame([{
        columna: 'TOTAL', 'fi': fi.sum(),
        'hi': round(hi.sum(), 4), '%': round(pct.sum(), 2),
        'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True)


def tabla_agrupada(df, columna):
    serie = df[columna].astype(float)
    n     = len(serie)
    vmin  = float(serie.min())
    vmax  = float(serie.max())
    rango = vmax - vmin

    k = math.ceil(1 + 3.322 * math.log10(n))
    c = math.ceil(rango / k)

    limites = sorted(set([round(vmin + i * c, 4) for i in range(k + 1)]))
    if limites[-1] < vmax:
        limites[-1] = vmax + 0.001

    k_real = len(limites) - 1
    bins   = pd.cut(serie, bins=limites, right=False, include_lowest=True)
    fi     = bins.value_counts().sort_index()
    hi     = (fi / n).round(4)
    pct    = (hi * 100).round(2)
    Fi     = fi.cumsum()
    Hi     = hi.cumsum().round(4)
    xi     = [round((iv.left + iv.right) / 2, 2) for iv in fi.index]
    etiq   = [f"[{iv.left:.1f} – {iv.right:.1f})" for iv in fi.index]

    tabla = pd.DataFrame({
        'Intervalo'  : etiq,
        'xi (Marca)' : xi,
        'fi'         : fi.values,
        'hi'         : hi.values,
        '%'          : pct.values,
        'Fi'         : Fi.values,
        'Hi'         : Hi.values,
    })
    params = {
        'n': n, 'min': vmin, 'max': vmax,
        'rango': round(rango, 4), 'k': k_real, 'c': c,
        'xi': xi, 'fi': fi.values, 'limites': limites,
    }
    total = pd.DataFrame([{
        'Intervalo': 'TOTAL', 'xi (Marca)': '—',
        'fi': fi.sum(), 'hi': round(hi.sum(), 4),
        '%' : round(pct.sum(), 2), 'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True), params


def calcular_medidas(serie_raw):
    serie = serie_raw.astype(float).dropna().values
    n     = len(serie)

    media   = float(np.mean(serie))
    mediana = float(np.median(serie))

    res_moda = stats.mode(serie, keepdims=True)
    if res_moda.count[0] == 1 and len(np.unique(serie)) == n:
        moda_str = "Sin moda"
        moda_val = None
    else:
        moda_val = float(res_moda.mode[0])
        moda_str = f"{moda_val:.4f}"

    mg = float(stats.gmean(serie)) if np.all(serie > 0) else None
    ma = float(stats.hmean(serie)) if np.all(serie > 0) else None
    mc = float(np.sqrt(np.mean(serie ** 2)))

    q1  = float(np.percentile(serie, 25))
    q2  = float(np.percentile(serie, 50))
    q3  = float(np.percentile(serie, 75))
    d5  = q2
    p10 = float(np.percentile(serie, 10))
    p75 = q3
    p90 = float(np.percentile(serie, 90))

    rango   = float(np.max(serie) - np.min(serie))
    desv_m  = float(np.mean(np.abs(serie - media)))
    var_pob = float(np.var(serie, ddof=0))
    var_mue = float(np.var(serie, ddof=1))
    desv_e  = float(np.std(serie, ddof=1))
    cv      = float((desv_e / media) * 100) if media != 0 else None

    caf = float(stats.skew(serie))

    if moda_val is not None:
        cap = (media - moda_val) / desv_e
    else:
        cap = 3.0 * (media - mediana) / desv_e

    denom_bow = q3 - q1
    cab = (q3 + q1 - 2.0 * q2) / denom_bow if denom_bow != 0 else 0.0

    curtosis = float(stats.kurtosis(serie))

    denom_kb = 2.0 * (p90 - p10)
    k_bow    = float((q3 - q1) / denom_kb) if denom_kb != 0 else None

    return {
        'n': n, 'media': media, 'mediana': mediana,
        'moda_str': moda_str, 'moda_val': moda_val,
        'mg': mg, 'ma': ma, 'mc': mc,
        'q1': q1, 'q2': q2, 'q3': q3,
        'd5': d5, 'p10': p10, 'p75': p75, 'p90': p90,
        'rango': rango, 'desv_m': desv_m,
        'var_pob': var_pob, 'var_mue': var_mue,
        'desv_e': desv_e, 'cv': cv,
        'caf': caf, 'cap': cap, 'cab': cab,
        'curtosis': curtosis, 'k_bow': k_bow,
    }


def mediana_agrupada(df, columna):
    tabla, params = tabla_agrupada(df, columna)
    datos   = tabla[tabla['Intervalo'] != 'TOTAL'].reset_index(drop=True)
    n       = params['n']
    fi_vals = np.array(datos['fi'].values, dtype=float)
    c_amp   = float(params['c'])
    Fi_vals = np.cumsum(fi_vals)

    objetivo  = n / 2.0
    clase_idx = int(np.searchsorted(Fi_vals, objetivo, side='left'))
    clase_idx = min(clase_idx, len(fi_vals) - 1)

    Fi_ant   = float(Fi_vals[clase_idx - 1]) if clase_idx > 0 else 0.0
    fi_clase = float(fi_vals[clase_idx])

    intervalo_str = datos.iloc[clase_idx]['Intervalo']
    Li = float(intervalo_str.split('–')[0].replace('[', '').strip())

    me_val = Li + ((objetivo - Fi_ant) / fi_clase) * c_amp

    return {
        'me_val'       : round(me_val, 4),
        'intervalo_str': intervalo_str,
        'Li'           : Li,
        'N_2'          : objetivo,
        'Fi_ant'       : Fi_ant,
        'fi_clase'     : fi_clase,
        'c_amp'        : c_amp,
    }


def interp_asimetria(valor):
    if abs(valor) < 0.05:
        return "La distribución es simétrica.\nLos datos se reparten de forma\nequilibrada alrededor del promedio."
    elif valor > 0:
        return ("La distribución se inclina hacia la derecha.\n"
                "Hay más datos bajos y algunos valores\n"
                "muy altos que jalan la cola.\n"
                "El promedio queda por encima de la mediana.")
    else:
        return ("La distribución se inclina hacia la izquierda.\n"
                "Hay más datos altos y algunos valores\n"
                "muy bajos que jalan la cola.\n"
                "El promedio queda por debajo de la mediana.")


def interp_curtosis(k):
    if abs(k) < 0.3:
        return ("Distribución normal — los datos\n"
                "se concentran de forma parecida\n"
                "a la curva de campana estándar.")
    elif k > 0:
        return ("Distribución apuntada — hay una\n"
                "concentración alta de datos cerca\n"
                "del promedio, con colas más pesadas.")
    else:
        return ("Distribución aplanada — los datos\n"
                "están más dispersos y la curva\n"
                "es más chata que la normal.")


def interp_cv(cv):
    if cv < 15:
        return "Grupo homogéneo — poca variabilidad"
    elif cv < 30:
        return "Variabilidad moderada"
    else:
        return "Grupo heterogéneo — alta variabilidad"


def interp_desviacion(desv, media, variable):
    return (f"En promedio, cada estudiante\n"
            f"se aleja ±{desv:.2f} puntos del\n"
            f"promedio general de {media:.2f}.")


def g1_barras_carrera(df, ax):
    conteo = df['carrera'].value_counts().sort_index()
    barras = ax.bar(conteo.index, conteo.values,
                    color=COLORES['barras'][:len(conteo)],
                    edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(barras, conteo.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(val), ha='center', va='bottom',
                fontweight='bold', fontsize=11, color='#1F3864')
    ax.set_title('Estudiantes por Carrera')
    ax.set_xlabel('Carrera')
    ax.set_ylabel('Cantidad de estudiantes')
    ax.set_ylim(0, conteo.max() + 3)


def g2_baston_materias(df, ax):
    conteo = df['materias_aprobadas'].value_counts().sort_index()
    ax.vlines(conteo.index, 0, conteo.values,
              color=COLORES['barras'][0], linewidth=2.5)
    ax.plot(conteo.index, conteo.values, 'o',
            color=COLORES['linea'], markersize=9, zorder=5)
    ax.set_title('Materias Aprobadas por Estudiante')
    ax.set_xlabel('Número de materias aprobadas')
    ax.set_ylabel('Cantidad de estudiantes')
    ax.set_ylim(0, conteo.max() + 1)
    ax.set_xticks(conteo.index)


def g3_histograma_poligono(df, ax):
    tabla, params = tabla_agrupada(df, 'edad')
    datos = tabla[tabla['Intervalo'] != 'TOTAL']
    fi    = np.array(datos['fi'].values, dtype=float)
    xi    = np.array(datos['xi (Marca)'].values, dtype=float)
    c     = float(params['c'])

    ax.bar(xi, fi, width=c * 0.9,
           color=COLORES['hist'], alpha=0.75,
           edgecolor='white', linewidth=1.2, label='Histograma')

    xi_ext = np.concatenate([np.array([xi[0] - c]), xi, np.array([xi[-1] + c])])
    fi_ext = np.concatenate([np.array([0.0]), fi, np.array([0.0])])
    ax.plot(xi_ext, fi_ext, 'o-',
            color=COLORES['linea'], linewidth=2.2,
            markersize=6, zorder=5, label='Polígono')

    ax.set_title('Distribución de Edades')
    ax.set_xlabel('Edad (punto medio del intervalo)')
    ax.set_ylabel('Cantidad de estudiantes')
    ax.legend()


def g4_ojiva(df, ax):
    tabla, _ = tabla_agrupada(df, 'edad')
    datos = tabla[tabla['Intervalo'] != 'TOTAL']
    xi    = np.array(datos['xi (Marca)'].values, dtype=float)
    Fi    = np.array(datos['Fi'].values, dtype=float)

    ax.plot(xi, Fi, 's-',
            color=COLORES['ojiva'], linewidth=2.5,
            markersize=8, markerfacecolor='white',
            markeredgewidth=2.5, label='Acumulado')
    ax.axhline(y=len(df), color='gray', linestyle='--',
               alpha=0.5, linewidth=1.2, label=f'Total: {len(df)}')
    ax.fill_between(xi, Fi, alpha=0.08, color=COLORES['ojiva'])
    ax.set_title('Edades Acumuladas (Ojiva)')
    ax.set_xlabel('Edad (punto medio del intervalo)')
    ax.set_ylabel('Estudiantes acumulados')
    ax.set_ylim(0, len(df) + 3)
    ax.legend()


def g5_torta_carrera(df, ax):
    conteo = df['carrera'].value_counts().sort_index()
    wedges, texts, autotexts = ax.pie(
        conteo.values,
        labels=conteo.index,
        colors=COLORES['torta'],
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        textprops={'fontsize': 10},
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_color('white')
        at.set_fontsize(11)
    ax.set_title('Proporción de Estudiantes por Carrera')


def g6_distribucion_desv(df, columna, ax):
    serie = df[columna].astype(float).values
    m     = calcular_medidas(df[columna])
    media = m['media']
    s     = m['desv_e']

    ax.hist(serie, bins='auto', color=COLORES['hist'],
            alpha=0.55, edgecolor='white', density=True, label='Datos')

    kde_x = np.linspace(serie.min() - 1, serie.max() + 1, 300)
    kde   = stats.gaussian_kde(serie)
    ax.plot(kde_x, kde(kde_x), color=COLORES['hist'],
            linewidth=2.5, label='Curva')

    ax.axvline(media,       color=COLORES['rojo'],  lw=2.0, ls='-',
               label=f'Promedio = {media:.2f}')
    ax.axvline(media - s,   color=COLORES['linea'], lw=1.8, ls='--',
               label=f'± 1 desv. = {s:.2f}')
    ax.axvline(media + s,   color=COLORES['linea'], lw=1.8, ls='--')
    ax.axvline(media - 2*s, color=COLORES['verde'], lw=1.5, ls=':',
               label=f'± 2 desv. = {2*s:.2f}')
    ax.axvline(media + 2*s, color=COLORES['verde'], lw=1.5, ls=':')

    nombre = columna.replace('_', ' ').title()
    ax.set_title(f'{nombre} — Promedio y dispersión')
    ax.set_xlabel(nombre)
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)


def _separador(parent, bg="#D0D8E8"):
    tk.Frame(parent, bg=bg, height=1).pack(fill='x', pady=4)


def _scrollable_frame(parent):
    canvas = tk.Canvas(parent, bg="#F0F2F8", highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
    inner = tk.Frame(canvas, bg="#F0F2F8")

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor='nw')
    canvas.configure(yscrollcommand=sb.set)

    canvas.pack(side='left', fill='both', expand=True)
    sb.pack(side='right', fill='y')

    def _on_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_wheel)

    return canvas, inner


def _render_seccion(parent, titulo, filas, bg_header, bg_par, bg_imp):
    box = tk.LabelFrame(
        parent, text=f"  {titulo}  ",
        font=("Arial", 10, "bold"),
        fg="#1F3864", bg=bg_header,
        relief='groove', bd=2,
    )
    box.pack(fill='x', padx=12, pady=6)

    for j, fila in enumerate(filas):
        etq       = fila[0]
        val       = fila[1]
        color_val = fila[2] if len(fila) > 2 else "#C00000"
        bg_r      = bg_par if j % 2 == 0 else bg_imp

        tk.Label(box, text=etq,
                 font=("Arial", 9, "bold"),
                 fg="#1F3864", bg=bg_r,
                 anchor='w', width=32,
                 ).grid(row=j, column=0, sticky='ew', padx=6, pady=2)
        tk.Label(box, text=str(val),
                 font=("Arial", 10, "bold"),
                 fg=color_val, bg=bg_r,
                 anchor='center', width=22,
                 ).grid(row=j, column=1, sticky='ew', padx=6, pady=2)
    return box


def _render_interpretacion(parent, texto, bg="#FFF3E0", fg="#5A3800"):
    box = tk.LabelFrame(
        parent, text="  Lo que esto significa  ",
        font=("Arial", 9, "bold"),
        fg="#5A3800", bg=bg, relief='groove', bd=1,
    )
    box.pack(fill='x', padx=12, pady=4)
    tk.Label(box, text=texto,
             font=("Arial", 10),
             fg=fg, bg=bg,
             justify='left', anchor='w',
             ).pack(padx=10, pady=6, anchor='w')
    return box


class App(tk.Tk):

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.title("Laboratorio 1 — Estadística Descriptiva con Python")
        self.geometry("1340x860")
        self.configure(bg="#F0F2F8")
        self.resizable(True, True)
        self._encabezado()
        self._pestanas()

    def _encabezado(self):
        f = tk.Frame(self, bg="#1F3864", height=58)
        f.pack(fill='x')
        f.pack_propagate(False)
        tk.Label(f,
                 text="  📊  LABORATORIO 1 — ESTADÍSTICA DESCRIPTIVA CON PYTHON",
                 bg="#1F3864", fg="white",
                 font=("Arial", 14, "bold")).pack(side='left', padx=16, pady=12)
        tk.Label(f,
                 text=f"n = {len(self.df)} estudiantes  |  6 variables  |  3 carreras",
                 bg="#1F3864", fg="#BDD7EE",
                 font=("Arial", 10)).pack(side='right', padx=20)

    def _pestanas(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TNotebook', background='#F0F2F8', borderwidth=0)
        s.configure('TNotebook.Tab',
                    background='#D6E4F0', foreground='#1F3864',
                    font=('Arial', 10, 'bold'), padding=[16, 7])
        s.map('TNotebook.Tab',
              background=[('selected', '#2E75B6')],
              foreground=[('selected', 'white')])

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=10, pady=8)

        t1 = ttk.Frame(nb); nb.add(t1, text="  📋  Datos  ")
        t2 = ttk.Frame(nb); nb.add(t2, text="  📊  Tablas  ")
        t3 = ttk.Frame(nb); nb.add(t3, text="  📈  Gráficos  ")
        t4 = ttk.Frame(nb); nb.add(t4, text="  🔢  Tendencia Central  ")
        t5 = ttk.Frame(nb); nb.add(t5, text="  📏  Dispersión  ")
        t6 = ttk.Frame(nb); nb.add(t6, text="  📐  Asimetría y Curtosis  ")
        t7 = ttk.Frame(nb); nb.add(t7, text="  ⚙️  Sturges  ")

        self._tab_datos(t1)
        self._tab_tablas(t2)
        self._tab_graficos(t3)
        self._tab_tendencia_central(t4)
        self._tab_dispersion(t5)
        self._tab_asimetria_curtosis(t6)
        self._tab_params(t7)

    def _tab_datos(self, p):
        tk.Label(p, text="Datos cargados desde datos_estudiantes.csv",
                 font=("Arial", 11, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(10, 4))

        f = tk.Frame(p, bg="#F0F2F8")
        f.pack(fill='both', expand=True, padx=14, pady=6)

        s = ttk.Style()
        s.configure("DT.Treeview",
                     font=('Arial', 10), rowheight=26,
                     background="white", fieldbackground="white")
        s.configure("DT.Treeview.Heading",
                     font=('Arial', 10, 'bold'),
                     foreground="#1F3864", background="#DEEAF1")
        s.map("DT.Treeview", background=[('selected', '#2E75B6')])

        cols = list(self.df.columns)
        tree = ttk.Treeview(f, columns=cols, show='headings',
                             height=22, style="DT.Treeview")
        for col in cols:
            tree.heading(col, text=col.replace('_', ' ').title())
            tree.column(col, width=170, anchor='center')

        for i, row in self.df.iterrows():
            tag = 'par' if i % 2 == 0 else 'imp'
            tree.insert('', 'end', values=list(row), tags=(tag,))

        tree.tag_configure('par', background='#DEEAF1')
        tree.tag_configure('imp', background='white')

        sy = ttk.Scrollbar(f, orient='vertical',   command=tree.yview)
        sx = ttk.Scrollbar(f, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side='right',  fill='y')
        sx.pack(side='bottom', fill='x')
        tree.pack(fill='both', expand=True)

    def _tab_tablas(self, p):
        nb2 = ttk.Notebook(p)
        nb2.pack(fill='both', expand=True, padx=8, pady=8)

        subtabs = [
            ("Carrera",           tabla_cualitativa(self.df, 'carrera')),
            ("Género",            tabla_cualitativa(self.df, 'genero')),
            ("Materias",          tabla_discreta(self.df, 'materias_aprobadas')),
            ("Edad agrupada",     tabla_agrupada(self.df, 'edad')[0]),
            ("Promedio agrupado", tabla_agrupada(self.df, 'promedio_acumulado')[0]),
            ("Materias agrupada", tabla_agrupada(self.df, 'materias_aprobadas')[0]),
        ]
        for nombre, tabla in subtabs:
            fr = ttk.Frame(nb2)
            nb2.add(fr, text=f"  {nombre}  ")
            self._render_tabla(fr, tabla)

    def _render_tabla(self, parent, tabla):
        s = ttk.Style()
        s.configure("FT.Treeview",
                     font=('Arial', 10), rowheight=26)
        s.configure("FT.Treeview.Heading",
                     font=('Arial', 10, 'bold'), foreground="#1F3864",
                     background="#DEEAF1")

        cols = list(tabla.columns)
        tree = ttk.Treeview(parent, columns=cols, show='headings',
                             height=20, style="FT.Treeview")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor='center')

        for i, row in tabla.iterrows():
            es_total = str(row.iloc[0]).upper() == 'TOTAL'
            tag = 'tot' if es_total else ('par' if i % 2 == 0 else 'imp')
            tree.insert('', 'end', values=list(row), tags=(tag,))

        tree.tag_configure('par', background='#DEEAF1')
        tree.tag_configure('imp', background='white')
        tree.tag_configure('tot', background='#FFF2CC',
                            font=('Arial', 10, 'bold'))

        sb = ttk.Scrollbar(parent, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        tree.pack(fill='both', expand=True, padx=10, pady=10)

    def _tab_graficos(self, p):
        ctrl = tk.Frame(p, bg="#E8EEF8", height=48)
        ctrl.pack(fill='x')
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="  Selecciona el gráfico:",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#E8EEF8").pack(side='left', pady=12)

        opciones = [
            "1 — Barras: Estudiantes por Carrera",
            "2 — Bastón: Materias Aprobadas",
            "3 — Histograma + Polígono: Distribución de Edades",
            "4 — Ojiva: Edades Acumuladas",
            "5 — Torta: Proporción por Carrera",
            "🗂  Ver todos los gráficos",
        ]
        self.var_g = tk.StringVar(value=opciones[0])
        combo = ttk.Combobox(ctrl, textvariable=self.var_g,
                              values=opciones, state='readonly',
                              width=44, font=("Arial", 10))
        combo.pack(side='left', padx=8, pady=12)
        tk.Button(ctrl, text="  ▶  Generar  ",
                  bg="#2E75B6", fg="white",
                  font=("Arial", 10, "bold"),
                  relief='flat', cursor='hand2',
                  command=lambda: self._render_grafico(p)
                  ).pack(side='left', padx=6)

        self.canvas_frame = tk.Frame(p, bg="white")
        self.canvas_frame.pack(fill='both', expand=True)
        self._render_grafico(p)

    def _render_grafico(self, p):
        for w in self.canvas_frame.winfo_children():
            w.destroy()

        sel = self.var_g.get()

        if "todos" in sel.lower() or "Ver todos" in sel:
            fig = plt.figure(figsize=(16, 10), facecolor='#F4F6FA')
            fig.suptitle("Laboratorio 1 — Resumen de Gráficos",
                         fontsize=14, fontweight='bold',
                         color='#1F3864', y=0.98)
            gs = gridspec.GridSpec(2, 3, figure=fig,
                                   hspace=0.45, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.93, bottom=0.07)
            funcs = [g1_barras_carrera, g2_baston_materias,
                     g3_histograma_poligono, g4_ojiva, g5_torta_carrera]
            for i, fn in enumerate(funcs):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                fn(self.df, ax)
            ax_r = fig.add_subplot(gs[1, 2])
            ax_r.axis('off')
            ax_r.text(0.5, 0.5,
                      "Resumen del dataset\n\n"
                      f"Total estudiantes: {len(self.df)}\n"
                      "Variables analizadas: 6\n"
                      "Carreras: Sistemas, Civil,\n"
                      "          Industrial\n"
                      "Género: M / F\n"
                      "Edades: 18 a 27 años\n"
                      "Promedios: 55.5 a 94.0",
                      ha='center', va='center',
                      fontsize=10, color='#1F3864', fontweight='bold',
                      transform=ax_r.transAxes,
                      bbox=dict(boxstyle='round,pad=0.6',
                                facecolor='#DEEAF1',
                                edgecolor='#2E75B6', linewidth=1.5))
            ax_r.set_title("Datos generales", color='#1F3864', fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F4F6FA')
            fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.12)
            if   "1" in sel: g1_barras_carrera(self.df, ax)
            elif "2" in sel: g2_baston_materias(self.df, ax)
            elif "3" in sel: g3_histograma_poligono(self.df, ax)
            elif "4" in sel: g4_ojiva(self.df, ax)
            elif "5" in sel: g5_torta_carrera(self.df, ax)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    def _tab_tendencia_central(self, p):
        ctrl = tk.Frame(p, bg="#E8EEF8", height=46)
        ctrl.pack(fill='x')
        ctrl.pack_propagate(False)
        tk.Label(ctrl, text="  Analizar variable:",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#E8EEF8").pack(side='left', pady=10)

        self.var_tc = tk.StringVar(value='materias_aprobadas')
        combo = ttk.Combobox(ctrl, textvariable=self.var_tc,
                              values=['materias_aprobadas', 'edad',
                                      'promedio_acumulado'],
                              state='readonly', width=22,
                              font=("Arial", 10))
        combo.pack(side='left', padx=8)
        tk.Button(ctrl, text="  Calcular  ",
                  bg="#2E75B6", fg="white",
                  font=("Arial", 10, "bold"),
                  relief='flat', cursor='hand2',
                  command=lambda: self._render_tendencia_central(tc_body)
                  ).pack(side='left', padx=6)

        outer = tk.Frame(p, bg="#F0F2F8")
        outer.pack(fill='both', expand=True)
        _, tc_body = _scrollable_frame(outer)

        self._render_tendencia_central(tc_body)

    def _render_tendencia_central(self, parent):
        for w in parent.winfo_children():
            w.destroy()

        col    = self.var_tc.get()
        m      = calcular_medidas(self.df[col])
        nombre = col.replace('_', ' ').title()

        tk.Label(parent,
                 text=f"Medidas de Tendencia Central — {nombre}",
                 font=("Arial", 12, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(10, 2))
        _separador(parent)

        _render_seccion(parent,
            "Promedio, Mediana y Moda",
            [
                ("Promedio  (suma de todos / cantidad)",
                 f"{m['media']:.4f}", "#1F3864"),
                ("Mediana  (valor del centro al ordenar)",
                 f"{m['mediana']:.4f}", "#70AD47"),
                ("Moda  (el valor que más se repite)",
                 m['moda_str'], "#ED7D31"),
            ],
            COLOR_SECCION['central'], "#DEEAF1", "#EEF4FB"
        )

        mg_str = f"{m['mg']:.4f}" if m['mg'] else "No aplica"
        ma_str = f"{m['ma']:.4f}" if m['ma'] else "No aplica"
        _render_seccion(parent,
            "Otros tipos de promedio",
            [
                ("Media Geométrica  (raíz n-ésima del producto)",
                 mg_str, "#7030A0"),
                ("Media Armónica  (para razones y velocidades)",
                 ma_str, "#C00000"),
                ("Media Cuadrática  (raíz del promedio de cuadrados)",
                 f"{m['mc']:.4f}", "#2E75B6"),
            ],
            COLOR_SECCION['central'], "#DEEAF1", "#EEF4FB"
        )

        if m['mg'] and m['ma']:
            ok = m['ma'] <= m['mg'] <= m['media'] <= m['mc']
            rel_txt = (
                f"  Orden correcto entre los 4 promedios:\n"
                f"  Armónica ≤ Geométrica ≤ Aritmética ≤ Cuadrática\n"
                f"  {m['ma']:.4f}  ≤  {m['mg']:.4f}  ≤  {m['media']:.4f}  ≤  {m['mc']:.4f}\n"
                f"  {'✅ Se cumple correctamente' if ok else '⚠️ Revisar cálculo'}"
            )
            box = tk.LabelFrame(parent,
                                 text="  Verificación de orden entre promedios  ",
                                 font=("Arial", 9, "bold"),
                                 fg="#1F4B00", bg="#E8F5E9",
                                 relief='groove', bd=1)
            box.pack(fill='x', padx=12, pady=4)
            tk.Label(box, text=rel_txt,
                     font=("Courier New", 10, "bold"),
                     fg="#1F4B00", bg="#E8F5E9",
                     justify='left').pack(padx=10, pady=6, anchor='w')

        _separador(parent)

        iqr = m['q3'] - m['q1']
        _render_seccion(parent,
            "Cuartiles, Deciles y Percentiles",
            [
                ("Cuartil 1  — el 25% de los datos queda por debajo",
                 f"{m['q1']:.4f}", "#2E75B6"),
                ("Cuartil 2  — la mitad de los datos (= mediana)",
                 f"{m['q2']:.4f}", "#2E75B6"),
                ("Cuartil 3  — el 75% de los datos queda por debajo",
                 f"{m['q3']:.4f}", "#2E75B6"),
                ("Decil 5    — coincide con la mediana",
                 f"{m['d5']:.4f}", "#70AD47"),
                ("Percentil 10  — solo el 10% queda por debajo",
                 f"{m['p10']:.4f}", "#ED7D31"),
                ("Percentil 75  — igual al Cuartil 3",
                 f"{m['p75']:.4f}", "#ED7D31"),
                ("Percentil 90  — solo el 10% queda por encima",
                 f"{m['p90']:.4f}", "#C00000"),
                ("Rango intercuartílico  (Q3 menos Q1)",
                 f"{iqr:.4f}", "#7030A0"),
            ],
            COLOR_SECCION['posicion'], "#E2EFDA", "#F0F8EB"
        )

        _separador(parent)

        info_me = mediana_agrupada(self.df, col)

        box_me = tk.LabelFrame(parent,
                                text="  Mediana calculada con datos agrupados  ",
                                font=("Arial", 10, "bold"),
                                fg="#1F3864", bg="#FFF8E7",
                                relief='groove', bd=2)
        box_me.pack(fill='x', padx=12, pady=6)

        detalle = (
            f"  Se trabaja con los datos organizados en intervalos.\n"
            f"  La mediana cae dentro del intervalo  {info_me['intervalo_str']}\n\n"
            f"  Límite inferior del intervalo  :  {info_me['Li']}\n"
            f"  La mitad de {int(info_me['N_2']*2)} datos es           :  {info_me['N_2']}\n"
            f"  Datos acumulados antes         :  {info_me['Fi_ant']}\n"
            f"  Datos dentro del intervalo     :  {info_me['fi_clase']}\n"
            f"  Amplitud del intervalo         :  {info_me['c_amp']}\n"
            f"  {'─'*50}\n"
            f"  Mediana (agrupada)  =  {info_me['Li']} + "
            f"(({info_me['N_2']} - {info_me['Fi_ant']}) / {info_me['fi_clase']}) "
            f"x {info_me['c_amp']}\n"
            f"                      =  {info_me['me_val']}"
        )
        tk.Label(box_me, text=detalle,
                 font=("Courier New", 10),
                 fg="#5A3800", bg="#FFF8E7",
                 justify='left').pack(padx=12, pady=8, anchor='w')

    def _tab_dispersion(self, p):
        ctrl = tk.Frame(p, bg="#E8EEF8", height=46)
        ctrl.pack(fill='x')
        ctrl.pack_propagate(False)
        tk.Label(ctrl, text="  Analizar variable:",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#E8EEF8").pack(side='left', pady=10)

        self.var_disp = tk.StringVar(value='materias_aprobadas')
        combo = ttk.Combobox(ctrl, textvariable=self.var_disp,
                              values=['materias_aprobadas', 'edad',
                                      'promedio_acumulado'],
                              state='readonly', width=22,
                              font=("Arial", 10))
        combo.pack(side='left', padx=8)
        tk.Button(ctrl, text="  Calcular  ",
                  bg="#2E75B6", fg="white",
                  font=("Arial", 10, "bold"),
                  relief='flat', cursor='hand2',
                  command=lambda: self._render_dispersion(disp_body, gf)
                  ).pack(side='left', padx=6)

        panel = tk.Frame(p, bg="#F0F2F8")
        panel.pack(fill='both', expand=True)

        izq = tk.Frame(panel, bg="#F0F2F8", width=480)
        izq.pack(side='left', fill='y')
        izq.pack_propagate(False)

        der = tk.Frame(panel, bg="white")
        der.pack(side='left', fill='both', expand=True)

        outer = tk.Frame(izq, bg="#F0F2F8")
        outer.pack(fill='both', expand=True)
        _, disp_body = _scrollable_frame(outer)

        gf = der

        self._render_dispersion(disp_body, gf)

    def _render_dispersion(self, parent, gf):
        for w in parent.winfo_children():
            w.destroy()
        for w in gf.winfo_children():
            w.destroy()

        col    = self.var_disp.get()
        m      = calcular_medidas(self.df[col])
        nombre = col.replace('_', ' ').title()

        tk.Label(parent,
                 text=f"Qué tan dispersos están los datos — {nombre}",
                 font=("Arial", 12, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(10, 2))
        _separador(parent)

        _render_seccion(parent,
            "Medidas de dispersión absoluta",
            [
                ("Rango  (valor máximo menos el mínimo)",
                 f"{m['rango']:.4f}", "#1F3864"),
                ("Desviación media  (distancia promedio al promedio)",
                 f"{m['desv_m']:.4f}", "#2E75B6"),
                ("Varianza poblacional  (cuando se tiene todo el grupo)",
                 f"{m['var_pob']:.4f}", "#ED7D31"),
                ("Varianza muestral  (cuando se trabaja con una muestra)",
                 f"{m['var_mue']:.4f}", "#C00000"),
                ("Desviación estándar  (raíz de la varianza muestral)",
                 f"{m['desv_e']:.4f}", "#7030A0"),
            ],
            COLOR_SECCION['dispersion'], "#FCE4D6", "#FFF0E8"
        )

        cv_str = f"{m['cv']:.2f}%" if m['cv'] else "No disponible"
        _render_seccion(parent,
            "Coeficiente de Variación — permite comparar grupos distintos",
            [
                ("Coeficiente de Variación  (desv. estándar / promedio × 100)",
                 cv_str, "#C00000"),
                ("Promedio del grupo",
                 f"{m['media']:.4f}", "#1F3864"),
            ],
            COLOR_SECCION['dispersion'], "#FCE4D6", "#FFF0E8"
        )

        _separador(parent)
        tk.Label(parent,
                 text="¿Qué nos dice la desviación estándar?",
                 font=("Arial", 10, "bold"), fg="#5A3800",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(4, 2))

        interp_s = interp_desviacion(m['desv_e'], m['media'], col)
        if m['desv_e'] < m['media'] * 0.2:
            interp_s += "\n\nLa desviación es baja. Los datos están\nbastante concentrados cerca del promedio.\nEl grupo es homogéneo."
        else:
            interp_s += "\n\nLa desviación es considerable. Los datos\nse alejan bastante del promedio.\nHay variabilidad en el grupo."
        _render_interpretacion(parent, interp_s)

        if m['cv']:
            _render_interpretacion(
                parent,
                f"El coeficiente de variación es {m['cv']:.2f}%\n"
                f"{interp_cv(m['cv'])}\n\n"
                "Menos de 15%  → el grupo es homogéneo\n"
                "Entre 15-30%  → variabilidad moderada\n"
                "Más de 30%    → el grupo es heterogéneo",
                bg="#E8F5E9", fg="#1F4B00"
            )

        _separador(parent)
        tk.Label(parent,
                 text="Comparación entre las tres variables numéricas",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(4, 2))

        nombres_legibles = {
            'materias_aprobadas': 'Materias aprobadas',
            'edad': 'Edad',
            'promedio_acumulado': 'Promedio acumulado',
        }
        for vc in ['materias_aprobadas', 'edad', 'promedio_acumulado']:
            mc     = calcular_medidas(self.df[vc])
            cv_c   = f"{mc['cv']:.2f}%" if mc['cv'] else "N/A"
            color_cv = ("#C00000" if mc['cv'] and mc['cv'] > 30 else
                        "#ED7D31" if mc['cv'] and mc['cv'] > 15 else "#70AD47")
            row = tk.Frame(parent, bg="#DEEAF1" if vc == col else "#F0F2F8")
            row.pack(fill='x', padx=14, pady=1)
            tk.Label(row, text=nombres_legibles[vc],
                     font=("Arial", 9), fg="#1F3864",
                     bg=row.cget('bg'), width=24, anchor='w'
                     ).pack(side='left', padx=6)
            tk.Label(row,
                     text=f"Promedio={mc['media']:.2f}  Desv.={mc['desv_e']:.2f}  CV={cv_c}",
                     font=("Arial", 9, "bold"), fg=color_cv,
                     bg=row.cget('bg')
                     ).pack(side='left', padx=6)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#F4F6FA')
        fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.12)
        g6_distribucion_desv(self.df, col, ax)

        canvas = FigureCanvasTkAgg(fig, master=gf)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        plt.close(fig)

    def _tab_asimetria_curtosis(self, p):
        ctrl = tk.Frame(p, bg="#E8EEF8", height=46)
        ctrl.pack(fill='x')
        ctrl.pack_propagate(False)
        tk.Label(ctrl, text="  Analizar variable:",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#E8EEF8").pack(side='left', pady=10)

        self.var_ac = tk.StringVar(value='materias_aprobadas')
        combo = ttk.Combobox(ctrl, textvariable=self.var_ac,
                              values=['materias_aprobadas', 'edad',
                                      'promedio_acumulado'],
                              state='readonly', width=22,
                              font=("Arial", 10))
        combo.pack(side='left', padx=8)
        tk.Button(ctrl, text="  Calcular  ",
                  bg="#2E75B6", fg="white",
                  font=("Arial", 10, "bold"),
                  relief='flat', cursor='hand2',
                  command=lambda: self._render_asimetria(ac_body, ac_gf)
                  ).pack(side='left', padx=6)

        panel = tk.Frame(p, bg="#F0F2F8")
        panel.pack(fill='both', expand=True)

        izq = tk.Frame(panel, bg="#F0F2F8", width=500)
        izq.pack(side='left', fill='y')
        izq.pack_propagate(False)
        der = tk.Frame(panel, bg="white")
        der.pack(side='left', fill='both', expand=True)

        outer = tk.Frame(izq, bg="#F0F2F8")
        outer.pack(fill='both', expand=True)
        _, ac_body = _scrollable_frame(outer)
        ac_gf = der

        self._render_asimetria(ac_body, ac_gf)

    def _render_asimetria(self, parent, gf):
        for w in parent.winfo_children():
            w.destroy()
        for w in gf.winfo_children():
            w.destroy()

        col    = self.var_ac.get()
        m      = calcular_medidas(self.df[col])
        nombre = col.replace('_', ' ').title()

        tk.Label(parent,
                 text=f"¿Hacia dónde se inclina la distribución? — {nombre}",
                 font=("Arial", 12, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(10, 2))
        _separador(parent)

        color_caf = "#C00000" if m['caf'] > 0 else "#2E75B6"
        color_cap = "#C00000" if m['cap'] > 0 else "#2E75B6"
        color_cab = "#C00000" if m['cab'] > 0 else "#2E75B6"

        _render_seccion(parent,
            "Coeficientes de Asimetría — tres formas de medirla",
            [
                ("Fisher  (el más preciso, usa todos los datos)",
                 f"{m['caf']:.4f}", color_caf),
                ("Pearson  (compara promedio con moda o mediana)",
                 f"{m['cap']:.4f}", color_cap),
                ("Bowley  (usa cuartiles, no necesita promedio)",
                 f"{m['cab']:.4f}", color_cab),
            ],
            COLOR_SECCION['forma'], "#EDE7F6", "#F5F0FB"
        )

        _render_interpretacion(parent,
            f"Resultado Fisher = {m['caf']:.4f}\n\n"
            f"{interp_asimetria(m['caf'])}\n\n"
            f"Promedio = {m['media']:.4f}   "
            f"Mediana = {m['mediana']:.4f}   "
            f"Moda = {m['moda_str']}",
            bg="#FFF3E0", fg="#5A3800"
        )

        reglas = tk.LabelFrame(parent,
                                text="  Cómo interpretar el número que sale  ",
                                font=("Arial", 9, "bold"),
                                fg="#1F3864", bg="#DEEAF1",
                                relief='groove', bd=1)
        reglas.pack(fill='x', padx=12, pady=4)
        tk.Label(reglas,
                 text=(
                     "  Valor negativo  → los datos se inclinan hacia la izquierda\n"
                     "  Valor cero      → los datos están equilibrados (simétrico)\n"
                     "  Valor positivo  → los datos se inclinan hacia la derecha\n\n"
                     "  Bowley: se usa cuando no se puede calcular el promedio\n"
                     "  Pearson: requiere conocer la moda o la mediana\n"
                     "  Fisher: el más completo, considera cada dato individual"
                 ),
                 font=("Arial", 9), fg="#1F3864", bg="#DEEAF1",
                 justify='left').pack(padx=10, pady=6, anchor='w')

        _separador(parent)

        color_k  = "#7030A0" if m['curtosis'] > 0 else (
                   "#70AD47" if m['curtosis'] < 0 else "#2E75B6")
        k_bow_str = f"{m['k_bow']:.4f}" if m['k_bow'] else "No disponible"

        _render_seccion(parent,
            "Curtosis — ¿la curva es apuntada o aplanada?",
            [
                ("Curtosis de Fisher  (compara con la curva normal)",
                 f"{m['curtosis']:.4f}", color_k),
                ("Curtosis de Bowley  (basada en cuartiles y percentiles)",
                 k_bow_str, color_k),
            ],
            COLOR_SECCION['forma'], "#EDE7F6", "#F5F0FB"
        )

        _render_interpretacion(parent,
            f"Resultado curtosis = {m['curtosis']:.4f}\n\n"
            f"{interp_curtosis(m['curtosis'])}\n\n"
            "Valor cero  → forma normal (campana estándar)\n"
            "Positivo    → más apuntada que la normal\n"
            "Negativo    → más aplanada que la normal",
            bg="#EDE7F6", fg="#3A0F5C"
        )

        _separador(parent)
        tk.Label(parent,
                 text="Comparación de las tres variables",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=14, pady=(4, 2))

        nombres_legibles = {
            'materias_aprobadas': 'Materias aprobadas',
            'edad': 'Edad',
            'promedio_acumulado': 'Promedio acumulado',
        }
        encab = tk.Frame(parent, bg="#2E75B6")
        encab.pack(fill='x', padx=14)
        for txt, w in [("Variable", 22), ("Fisher", 9), ("Pearson", 9),
                       ("Bowley", 9), ("Curtosis", 10), ("Inclinación", 14)]:
            tk.Label(encab, text=txt,
                     font=("Arial", 9, "bold"), fg="white", bg="#2E75B6",
                     width=w, anchor='center'
                     ).pack(side='left', padx=2, pady=3)

        for i, vc in enumerate(['materias_aprobadas', 'edad', 'promedio_acumulado']):
            mc    = calcular_medidas(self.df[vc])
            tipo  = ("Derecha" if mc['caf'] > 0.05 else
                     "Izquierda" if mc['caf'] < -0.05 else "Simétrica")
            bg_r  = "#DEEAF1" if i % 2 == 0 else "#F0F2F8"
            row   = tk.Frame(parent, bg=bg_r)
            row.pack(fill='x', padx=14, pady=1)
            for txt, w in [
                (nombres_legibles[vc][:20], 22),
                (f"{mc['caf']:.3f}", 9),
                (f"{mc['cap']:.3f}", 9),
                (f"{mc['cab']:.3f}", 9),
                (f"{mc['curtosis']:.3f}", 10),
                (tipo, 14),
            ]:
                tk.Label(row, text=txt,
                         font=("Arial", 9), fg="#1F3864", bg=bg_r,
                         width=w, anchor='center'
                         ).pack(side='left', padx=2, pady=2)

        serie = self.df[col].astype(float).values
        fig, axes = plt.subplots(1, 2, figsize=(9, 5), facecolor='#F4F6FA')
        fig.subplots_adjust(left=0.09, right=0.97, top=0.88,
                            bottom=0.12, wspace=0.35)

        ax1 = axes[0]
        ax1.hist(serie, bins='auto', color=COLORES['hist'],
                 alpha=0.45, edgecolor='white', density=True)
        kde_x = np.linspace(serie.min() - 1, serie.max() + 1, 300)
        kde   = stats.gaussian_kde(serie)
        ax1.plot(kde_x, kde(kde_x), color=COLORES['hist'],
                 linewidth=2.5, label='Curva')
        ax1.axvline(m['media'],   color=COLORES['rojo'],  lw=2.0, ls='-',
                    label=f"Promedio={m['media']:.2f}")
        ax1.axvline(m['mediana'], color=COLORES['verde'], lw=2.0, ls='--',
                    label=f"Mediana={m['mediana']:.2f}")
        if m['moda_val']:
            ax1.axvline(m['moda_val'], color=COLORES['linea'], lw=2.0,
                        ls=':', label=f"Moda={m['moda_val']:.2f}")
        ax1.set_title(f"Distribución de {nombre}")
        ax1.set_xlabel(nombre)
        ax1.set_ylabel("Densidad")
        ax1.legend(fontsize=8)

        ax2 = axes[1]
        bp = ax2.boxplot(serie, patch_artist=True, widths=0.4,
                         medianprops={'color': COLORES['rojo'], 'linewidth': 2.5})
        bp['boxes'][0].set_facecolor(COLORES['hist'])
        bp['boxes'][0].set_alpha(0.5)
        ax2.set_title(
            f"Q1={m['q1']:.1f}   Mediana={m['q2']:.1f}   Q3={m['q3']:.1f}"
        )
        ax2.set_xticks([1])
        ax2.set_xticklabels([nombre], fontsize=9)
        ax2.set_ylabel("Valor")

        canvas = FigureCanvasTkAgg(fig, master=gf)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        plt.close(fig)

    def _tab_params(self, p):
        tk.Label(p,
                 text="Cómo se determinan los intervalos — Regla de Sturges",
                 font=("Arial", 12, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=16, pady=(12, 4))

        f = tk.Frame(p, bg="#F0F2F8")
        f.pack(fill='both', expand=True, padx=16, pady=8)

        nombres_legibles = {
            'edad': 'Edad',
            'promedio_acumulado': 'Promedio Acumulado',
            'materias_aprobadas': 'Materias Aprobadas',
        }

        for i, col in enumerate(['edad', 'promedio_acumulado', 'materias_aprobadas']):
            serie = self.df[col].astype(float)
            nn    = len(serie)
            vmin  = serie.min(); vmax = serie.max()
            rango = vmax - vmin
            k_raw = 1 + 3.322 * math.log10(nn)
            k     = math.ceil(k_raw)
            c     = math.ceil(rango / k)

            box = tk.LabelFrame(f,
                                 text=f"  {nombres_legibles[col]}  ",
                                 font=("Arial", 10, "bold"),
                                 fg="#2E75B6", bg="#DEEAF1",
                                 relief='groove', bd=2)
            box.grid(row=0, column=i, padx=12, pady=8, sticky='nsew')
            f.columnconfigure(i, weight=1)

            filas = [
                ("Total de datos",         str(nn)),
                ("Valor mínimo",           str(vmin)),
                ("Valor máximo",           str(vmax)),
                ("Rango  (máx menos mín)", str(round(rango, 2))),
                ("Número de clases k",     f"{round(k_raw,4)} → {k}"),
                ("Amplitud de cada clase", str(c)),
            ]
            for j, (etq, val) in enumerate(filas):
                bg_r = "#DEEAF1" if j % 2 == 0 else "#EEF4FB"
                tk.Label(box, text=etq, font=("Arial", 9, "bold"),
                         fg="#1F3864", bg=bg_r, anchor='w', width=24
                         ).grid(row=j, column=0, sticky='ew', padx=6, pady=3)
                tk.Label(box, text=val, font=("Arial", 10),
                         fg="#C00000", bg=bg_r, anchor='center', width=18
                         ).grid(row=j, column=1, sticky='ew', padx=6, pady=3)

        fbox = tk.LabelFrame(f,
                              text="  ¿Cómo funciona la Regla de Sturges?  ",
                              font=("Arial", 10, "bold"),
                              fg="#1F3864", bg="#FFF8E7",
                              relief='groove', bd=2)
        fbox.grid(row=1, column=0, columnspan=3,
                  padx=12, pady=12, sticky='ew')

        texto = (
            "  Sturges propone una fórmula para decidir cuántos intervalos usar al agrupar datos.\n\n"
            "  Número de clases:  k = 1 + 3.322 × log₁₀(n)\n"
            "  Amplitud:          c = redondear hacia arriba ( Rango / k )\n\n"
            "  Donde:\n"
            "     n    →  la cantidad total de datos\n"
            "     3.322  →  una constante matemática fija\n"
            "     Rango  →  el dato mayor menos el dato menor\n\n"
            "  Una tabla bien construida siempre cumple:\n"
            "     La suma de todas las frecuencias = n\n"
            "     La suma de todas las frecuencias relativas = 1\n"
            "     La suma de todos los porcentajes = 100%"
        )
        tk.Label(fbox, text=texto,
                 font=("Arial", 10), fg="#1F3864",
                 bg="#FFF8E7", justify='left', anchor='w'
                 ).pack(padx=14, pady=10, anchor='w')


if __name__ == "__main__":
    ruta = 'datos_estudiantes.csv'
    if not os.path.exists(ruta):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datos_estudiantes.csv')
    df  = cargar_datos(ruta)
    app = App(df)
    app.mainloop()
