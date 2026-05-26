

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")                         
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
import tkinter as tk
from tkinter import ttk

# ─────────────────────────────────────────────────────────────────────────────
# ESTILO GLOBAL DE MATPLOTLIB
# ─────────────────────────────────────────────────────────────────────────────
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
}

# =============================================================================
# FASE 1 — CARGA DE DATOS
# =============================================================================
def cargar_datos(ruta='datos_estudiantes.csv'):
    """
    Carga el CSV y limpia el nombre de columnas.
    El archivo puede tener BOM (ï»¿) en la primera columna; se corrige aquí.
    """
    df = pd.read_csv(ruta, encoding='latin1')
    df.columns = ['id', 'carrera', 'genero',
                  'materias_aprobadas', 'edad', 'promedio_acumulado']
    return df

# =============================================================================
# FASE 2 — TABLAS DE FRECUENCIA
# =============================================================================

def tabla_cualitativa(df, columna):
    """
    Tabla de frecuencia para variable CUALITATIVA (nominal).
    Usa value_counts() — equivalente al CONTAR.SI de Excel.
    Columnas: fi, hi, %, Fi, Hi
    """
    n   = len(df)
    fi  = df[columna].value_counts().sort_index()
    hi  = (fi / n).round(4)
    pct = (hi * 100).round(2)
    Fi  = fi.cumsum()
    Hi  = hi.cumsum().round(4)

    tabla = pd.DataFrame({
        columna.capitalize(): fi.index,
        'fi' : fi.values,
        'hi' : hi.values,
        '%'  : pct.values,
        'Fi' : Fi.values,
        'Hi' : Hi.values,
    })
    total = pd.DataFrame([{
        columna.capitalize(): 'TOTAL',
        'fi': fi.sum(), 'hi': round(hi.sum(), 4),
        '%' : round(pct.sum(), 2), 'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True)


def tabla_discreta(df, columna):
    """
    Tabla de frecuencia para variable CUANTITATIVA DISCRETA.
    Pocos valores → se cuenta valor a valor, sin agrupar.
    """
    n   = len(df)
    fi  = df[columna].value_counts().sort_index()
    hi  = (fi / n).round(4)
    pct = (hi * 100).round(2)
    Fi  = fi.cumsum()
    Hi  = hi.cumsum().round(4)

    tabla = pd.DataFrame({
        columna : fi.index,
        'fi'    : fi.values,
        'hi'    : hi.values,
        '%'     : pct.values,
        'Fi'    : Fi.values,
        'Hi'    : Hi.values,
    })
    total = pd.DataFrame([{
        columna: 'TOTAL', 'fi': fi.sum(),
        'hi': round(hi.sum(), 4), '%': round(pct.sum(), 2),
        'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True)


def tabla_agrupada(df, columna):
    """
    Tabla de frecuencia AGRUPADA para variable CUANTITATIVA CONTINUA.

    Pasos del algoritmo:
      1. Calcular n, mín, máx, rango
      2. Regla de Sturges: k = ceil(1 + 3.322 × log10(n))
      3. Amplitud: c = ceil(rango / k)
      4. pd.cut() segmenta los datos en los intervalos [Li, Ls)
      5. Se calculan fi, hi, %, Fi, Hi y Marca de Clase xi = (Li+Ls)/2
    """
    serie = df[columna].astype(float)
    n     = len(serie)
    vmin  = float(serie.min())
    vmax  = float(serie.max())
    rango = vmax - vmin

    k = math.ceil(1 + 3.322 * math.log10(n))
    c = math.ceil(rango / k)

    # Construir límites garantizando monotonía estricta
    limites = sorted(set([round(vmin + i * c, 4) for i in range(k + 1)]))
    if limites[-1] < vmax:
        limites[-1] = vmax + 0.001

    k_real = len(limites) - 1

    bins = pd.cut(serie, bins=limites, right=False, include_lowest=True)
    fi   = bins.value_counts().sort_index()
    hi   = (fi / n).round(4)
    pct  = (hi * 100).round(2)
    Fi   = fi.cumsum()
    Hi   = hi.cumsum().round(4)
    xi   = [round((iv.left + iv.right) / 2, 2) for iv in fi.index]
    etiq = [f"[{iv.left:.1f} – {iv.right:.1f})" for iv in fi.index]

    tabla = pd.DataFrame({
        'Intervalo'  : etiq,
        'xi (Marca)' : xi,
        'fi'         : fi.values,
        'hi'         : hi.values,
        '%'          : pct.values,
        'Fi'         : Fi.values,
        'Hi'         : Hi.values,
    })
    params = {'n': n, 'min': vmin, 'max': vmax,
              'rango': rango, 'k': k_real, 'c': c}
    total = pd.DataFrame([{
        'Intervalo': 'TOTAL', 'xi (Marca)': '—',
        'fi': fi.sum(), 'hi': round(hi.sum(), 4),
        '%' : round(pct.sum(), 2), 'Fi': '—', 'Hi': '—',
    }])
    return pd.concat([tabla, total], ignore_index=True), params

# =============================================================================
# FASE 3 — FUNCIONES DE GRÁFICOS
# =============================================================================

def g1_barras_carrera(df, ax):
    """Gráfico 1 — Barras verticales: Frecuencia por Carrera (Cualitativa)"""
    conteo = df['carrera'].value_counts().sort_index()
    barras = ax.bar(conteo.index, conteo.values,
                    color=COLORES['barras'][:len(conteo)],
                    edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(barras, conteo.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(val), ha='center', va='bottom',
                fontweight='bold', fontsize=11, color='#1F3864')
    ax.set_title('Gráfico 1 — Frecuencia por Carrera')
    ax.set_xlabel('Carrera')
    ax.set_ylabel('Frecuencia (fi)')
    ax.set_ylim(0, conteo.max() + 3)


def g2_baston_materias(df, ax):
    """Gráfico 2 — Bastón: Materias Aprobadas (Cuantitativa Discreta)"""
    conteo = df['materias_aprobadas'].value_counts().sort_index()
    ax.vlines(conteo.index, 0, conteo.values,
              color=COLORES['barras'][0], linewidth=2.5)
    ax.plot(conteo.index, conteo.values, 'o',
            color=COLORES['linea'], markersize=9, zorder=5)
    ax.set_title('Gráfico 2 — Materias Aprobadas (Bastón)')
    ax.set_xlabel('N° de Materias Aprobadas')
    ax.set_ylabel('Frecuencia (fi)')
    ax.set_ylim(0, conteo.max() + 1)
    ax.set_xticks(conteo.index)


def g3_histograma_poligono(df, ax):
    """
    Gráfico 3 — Histograma + Polígono de Frecuencias: Edad (Agrupada)
    El polígono conecta las marcas de clase (xi) con sus frecuencias.
    Se agregan puntos cero al inicio y al final para cerrar el polígono.
    """
    tabla, params = tabla_agrupada(df, 'edad')
    datos = tabla[tabla['Intervalo'] != 'TOTAL']
    fi = datos['fi'].values.astype(float)
    xi = datos['xi (Marca)'].values.astype(float)
    c  = float(params['c'])

    ax.bar(xi, fi, width=c * 0.9,
           color=COLORES['hist'], alpha=0.75,
           edgecolor='white', linewidth=1.2, label='Histograma')

    xi_ext = np.concatenate([[xi[0] - c], xi, [xi[-1] + c]]) # type: ignore
    fi_ext = np.concatenate([[0], fi, [0]]) # type: ignore
    ax.plot(xi_ext, fi_ext, 'o-',
            color=COLORES['linea'], linewidth=2.2,
            markersize=6, zorder=5, label='Polígono')

    ax.set_title('Gráfico 3 — Histograma + Polígono (Edad)')
    ax.set_xlabel('Marca de Clase (xi)')
    ax.set_ylabel('Frecuencia (fi)')
    ax.legend()


def g4_ojiva(df, ax):
    """
    Gráfico 4 — Ojiva (Frecuencia Absoluta Acumulada): Edad
    Muestra cómo se acumula el total de estudiantes clase a clase.
    """
    tabla, _ = tabla_agrupada(df, 'edad')
    datos = tabla[tabla['Intervalo'] != 'TOTAL']
    xi = datos['xi (Marca)'].values.astype(float)
    Fi = datos['Fi'].values.astype(float)

    ax.plot(xi, Fi, 's-',
            color=COLORES['ojiva'], linewidth=2.5,
            markersize=8, markerfacecolor='white',
            markeredgewidth=2.5, label='Fi acumulada')
    ax.axhline(y=len(df), color='gray', linestyle='--',
               alpha=0.5, linewidth=1.2, label=f'n = {len(df)}')
    ax.fill_between(xi, Fi, alpha=0.08, color=COLORES['ojiva'])
    ax.set_title('Gráfico 4 — Ojiva (Frec. Acumulada — Edad)')
    ax.set_xlabel('Marca de Clase (xi)')
    ax.set_ylabel('Frecuencia Acumulada (Fi)')
    ax.set_ylim(0, len(df) + 3)
    ax.legend()


def g5_torta_carrera(df, ax):
    """Gráfico 5 — Torta / Circular: Distribución % por Carrera"""
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
    ax.set_title('Gráfico 5 — Distribución por Carrera (%)')

# =============================================================================
# FASE 4 — INTERFAZ GRÁFICA TKINTER
# =============================================================================

class App(tk.Tk):
    """
    Ventana principal con 4 pestañas:
      📋 Datos       — tabla con los 25 registros originales
      📊 Tablas      — sub-pestañas con cada tabla de frecuencia
      📈 Gráficos    — selector + canvas con los 5 gráficos
      ⚙️  Parámetros  — parámetros de Sturges por variable
    """

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.title("Laboratorio 1 — Estadística Descriptiva con Python")
        self.geometry("1280x800")
        self.configure(bg="#F0F2F8")
        self.resizable(True, True)
        self._encabezado()
        self._pestanas()

    # ── encabezado ────────────────────────────────────────────────────────
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

    # ── contenedor de pestañas ────────────────────────────────────────────
    def _pestanas(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TNotebook', background='#F0F2F8', borderwidth=0)
        s.configure('TNotebook.Tab',
                    background='#D6E4F0', foreground='#1F3864',
                    font=('Arial', 10, 'bold'), padding=[18, 8])
        s.map('TNotebook.Tab',
              background=[('selected', '#2E75B6')],
              foreground=[('selected', 'white')])

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=10, pady=8)

        t1 = ttk.Frame(nb); nb.add(t1, text="  📋  Datos Originales  ")
        t2 = ttk.Frame(nb); nb.add(t2, text="  📊  Tablas de Frecuencia  ")
        t3 = ttk.Frame(nb); nb.add(t3, text="  📈  Gráficos  ")
        t4 = ttk.Frame(nb); nb.add(t4, text="  ⚙️   Parámetros Sturges  ")

        self._tab_datos(t1)
        self._tab_tablas(t2)
        self._tab_graficos(t3)
        self._tab_params(t4)

    # ── TAB 1: DATOS ORIGINALES ───────────────────────────────────────────
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

    # ── TAB 2: TABLAS DE FRECUENCIA ───────────────────────────────────────
    def _tab_tablas(self, p):
        nb2 = ttk.Notebook(p)
        nb2.pack(fill='both', expand=True, padx=8, pady=8)

        subtabs = [
            ("Carrera (Cualit.)",     tabla_cualitativa(self.df, 'carrera')),
            ("Género (Cualit.)",      tabla_cualitativa(self.df, 'genero')),
            ("Materias (Discreta)",   tabla_discreta(self.df, 'materias_aprobadas')),
            ("Edad (Agrupada)",       tabla_agrupada(self.df, 'edad')[0]),
            ("Promedio (Agrupada)",   tabla_agrupada(self.df, 'promedio_acumulado')[0]),
            ("Mat. Aprobadas (Agrup)",tabla_agrupada(self.df, 'materias_aprobadas')[0]),
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

    # ── TAB 3: GRÁFICOS ───────────────────────────────────────────────────
    def _tab_graficos(self, p):
        # Barra de control
        ctrl = tk.Frame(p, bg="#E8EEF8", height=48)
        ctrl.pack(fill='x')
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="  Selecciona el gráfico:",
                 font=("Arial", 10, "bold"), fg="#1F3864",
                 bg="#E8EEF8").pack(side='left', pady=12)

        opciones = [
            "1 — Barras: Frecuencia por Carrera",
            "2 — Bastón: Materias Aprobadas",
            "3 — Histograma + Polígono: Edad",
            "4 — Ojiva: Frecuencia Acumulada Edad",
            "5 — Torta: Distribución por Carrera (%)",
            "🗂  Todos los gráficos",
        ]
        self.var_g = tk.StringVar(value=opciones[0])
        combo = ttk.Combobox(ctrl, textvariable=self.var_g,
                              values=opciones, state='readonly',
                              width=36, font=("Arial", 10))
        combo.pack(side='left', padx=8, pady=12)

        tk.Button(ctrl, text="  ▶  Generar  ",
                  bg="#2E75B6", fg="white",
                  font=("Arial", 10, "bold"),
                  relief='flat', cursor='hand2',
                  command=lambda: self._render_grafico(p)
                  ).pack(side='left', padx=6)

        self.canvas_frame = tk.Frame(p, bg="white")
        self.canvas_frame.pack(fill='both', expand=True)

        self._render_grafico(p)   # gráfico inicial al abrir

    def _render_grafico(self, p):
        for w in self.canvas_frame.winfo_children():
            w.destroy()

        sel = self.var_g.get()

        if "Todos" in sel:
            fig = plt.figure(figsize=(16, 10), facecolor='#F4F6FA')
            fig.suptitle("Laboratorio 1 — Todos los Gráficos",
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
            # Panel resumen
            ax_r = fig.add_subplot(gs[1, 2])
            ax_r.axis('off')
            ax_r.text(0.5, 0.5,
                      "Resumen del Dataset\n\n"
                      f"n = {len(self.df)} estudiantes\n"
                      "Variables: 6\nCarreras: 3\n"
                      "Género: M / F\nEdad: 18 – 27\n"
                      "Promedio: 55.5 – 94.0",
                      ha='center', va='center',
                      fontsize=11, color='#1F3864',
                      fontweight='bold',
                      transform=ax_r.transAxes,
                      bbox=dict(boxstyle='round,pad=0.6',
                                facecolor='#DEEAF1',
                                edgecolor='#2E75B6',
                                linewidth=1.5))
            ax_r.set_title("Dataset", color='#1F3864', fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F4F6FA')
            fig.subplots_adjust(left=0.10, right=0.95,
                                top=0.90, bottom=0.12)
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

    # ── TAB 4: PARÁMETROS STURGES ─────────────────────────────────────────
    def _tab_params(self, p):
        tk.Label(p,
                 text="⚙️  Parámetros de Agrupación — Regla de Sturges",
                 font=("Arial", 12, "bold"), fg="#1F3864",
                 bg="#F0F2F8").pack(anchor='w', padx=16, pady=(12, 4))

        f = tk.Frame(p, bg="#F0F2F8")
        f.pack(fill='both', expand=True, padx=16, pady=8)

        vars_num = ['edad', 'promedio_acumulado', 'materias_aprobadas']

        for i, col in enumerate(vars_num):
            serie = self.df[col].astype(float)
            nn    = len(serie)
            vmin  = serie.min(); vmax = serie.max()
            rango = vmax - vmin
            k_raw = 1 + 3.322 * math.log10(nn)
            k     = math.ceil(k_raw)
            c     = math.ceil(rango / k)

            box = tk.LabelFrame(f,
                                 text=f"  Variable: {col.replace('_',' ').title()}  ",
                                 font=("Arial", 10, "bold"),
                                 fg="#2E75B6", bg="#DEEAF1",
                                 relief='groove', bd=2)
            box.grid(row=0, column=i, padx=12, pady=8, sticky='nsew')
            f.columnconfigure(i, weight=1)

            filas = [
                ("n (total datos)",        str(nn)),
                ("Mínimo",                 str(vmin)),
                ("Máximo",                 str(vmax)),
                ("Rango  R = máx − mín",   str(round(rango, 2))),
                ("k = ceil(1+3.322×log n)", f"{round(k_raw,4)} → {k}"),
                ("Amplitud  c = ⌈R/k⌉",    str(c)),
            ]
            for j, (etq, val) in enumerate(filas):
                bg_r = "#DEEAF1" if j % 2 == 0 else "#EEF4FB"
                tk.Label(box, text=etq, font=("Arial", 9, "bold"),
                         fg="#1F3864", bg=bg_r,
                         anchor='w', width=24
                         ).grid(row=j, column=0, sticky='ew', padx=6, pady=3)
                tk.Label(box, text=val, font=("Arial", 10),
                         fg="#C00000", bg=bg_r,
                         anchor='center', width=18
                         ).grid(row=j, column=1, sticky='ew', padx=6, pady=3)

        # Fórmula explicada
        fbox = tk.LabelFrame(f,
                              text="  📐  Fórmula y Propiedades  ",
                              font=("Arial", 10, "bold"),
                              fg="#1F3864", bg="#FFF8E7",
                              relief='groove', bd=2)
        fbox.grid(row=1, column=0, columnspan=3,
                  padx=12, pady=12, sticky='ew')

        texto = (
            "  k  =  1  +  3.322  ×  log₁₀(n)\n\n"
            "  k      →  Número de clases recomendadas\n"
            "  n      →  Total de datos\n"
            "  3.322  →  Constante de Sturges  (= 1 / log₁₀(2))\n\n"
            "  Amplitud :  c  =  ⌈ Rango / k ⌉   (redondeo hacia arriba)\n"
            "  Rango    :  R  =  Máximo − Mínimo\n\n"
            "  Propiedades :   Σ fi = n   |   Σ hi = 1.0000   |   Σ % = 100.00%"
        )
        tk.Label(fbox, text=texto,
                 font=("Courier New", 10), fg="#1F3864",
                 bg="#FFF8E7", justify='left', anchor='w'
                 ).pack(padx=14, pady=10, anchor='w')
if __name__ == "__main__":
    ruta = 'datos_estudiantes.csv'
    if not os.path.exists(ruta):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datos_estudiantes.csv')
    df = cargar_datos(ruta)
    app = App(df)
    app.mainloop()
