import os                                                        # os: maneja carpetas y archivos (crear ./figuras, unir rutas)
import sys                                                       # sys: lee lo que se escribe al ejecutar (ej. 'figuras')
import tkinter as tk                                             # tkinter: crea la ventana, botones, barras y casillas
from tkinter import ttk, filedialog, messagebox                  # ttk: pestañas · filedialog: 'Abrir foto' · messagebox: avisos de error

import numpy as np                                               # numpy: matrices (la imagen) y transformada de Fourier (np.fft)
import matplotlib.pyplot as plt                                  # matplotlib: dibuja las gráficas que se guardan en ./figuras
from matplotlib.figure import Figure                             # Figure: el lienzo donde se dibujan las 3 imágenes de la ventana
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # pone ese lienzo de matplotlib dentro de la ventana de tkinter
from PIL import Image                                            # PIL (Pillow): abre fotos jpg/png y las pasa a escala de grises
from scipy.ndimage import gaussian_filter                        # scipy: desenfoque gaussiano, para comprobar que calor = gaussiana
from skimage import data, transform                              # scikit-image: data trae la foto de la Luna · transform cambia su tamaño


# PARTE 1 · TEMA 1: ecuación del calor por diferencias finitas
def laplaciano(u, frontera):
    if frontera == "periodica":
        return (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
                np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
    if frontera == "dirichlet":
        p = np.pad(u, 1, mode="constant", constant_values=0.0)
    elif frontera == "neumann":
        p = np.pad(u, 1, mode="edge")
    else:
        raise ValueError(frontera)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * u


def calor_diferencias(u0, k, t, frontera="periodica", r=0.2):
    dt = r / k
    pasos = int(round(t / dt))
    u = u0.copy()
    for _ in range(pasos):
        u = u + r * laplaciano(u, frontera)
    return u, pasos


# PARTE 2 · TEMA 2: solución por transformada de Fourier
def frecuencias(forma):
    wy = 2 * np.pi * np.fft.fftfreq(forma[0])
    wx = 2 * np.pi * np.fft.fftfreq(forma[1])
    WX, WY = np.meshgrid(wx, wy)
    return WX**2 + WY**2


def calor_fourier(u0, k, t):
    U0 = np.fft.fft2(u0)
    H = np.exp(-k * frecuencias(u0.shape) * t)
    return np.real(np.fft.ifft2(U0 * H))


# PARTE 3 · TEMA 3: calor hacia atrás y filtro de Wiener
def calor_atras(img, H):
    return np.real(np.fft.ifft2(np.fft.fft2(img) / H))


def wiener(img, H, lam):
    G = H / (H**2 + lam)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * G))


# PARTE 4 · Herramientas: imágenes de prueba y medida de calidad
def imagen_luna(n=256):
    img = data.moon().astype(float) / 255.0
    return transform.resize(img, (n, n), anti_aliasing=True)


def imagen_placa(alto=96, ancho=320):
    fig = plt.figure(figsize=(ancho / 100, alto / 100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.5, "UCO 2026", fontsize=40, weight="bold",
             ha="center", va="center", family="DejaVu Sans")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=float)[:, :, :3] / 255.0
    plt.close(fig)
    img = rgba.mean(axis=2)
    img[:4, :] = img[-4:, :] = 0
    img[:, :4] = img[:, -4:] = 0
    return img


def mse(a, b):
    return np.mean((a - b) ** 2)


def psnr(a, b):
    return 10 * np.log10(1.0 / mse(a, b))


# PARTE 5 · Gráficas y números para el informe
CARPETA = "figuras"
rng = np.random.default_rng(seed=2026)
resultados = []


def reportar(texto):
    print(texto)
    resultados.append(texto)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, nombre), dpi=150)
    plt.close(fig)


def mostrar(ax, img, titulo):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")


def generar_figuras():
    plt.switch_backend("Agg")
    os.makedirs(CARPETA, exist_ok=True)

    # ACTO 1: el calor hacia adelante
    reportar("=" * 70)
    reportar("ACTO 1: calor hacia adelante")
    reportar("=" * 70)

    luna = imagen_luna()
    k, t = 1.0, 4.0
    sigma = np.sqrt(2 * k * t)

    u_df, pasos = calor_diferencias(luna, k, t, "periodica")
    u_ff = calor_fourier(luna, k, t)
    u_gs = gaussian_filter(luna, sigma=sigma, mode="wrap")

    reportar(f"k = {k}, t = {t}  ->  sigma equivalente = sqrt(2kt) = {sigma:.3f} px")
    reportar(f"Diferencias finitas: {pasos} pasos con r = k·dt/dx² = 0.2")
    reportar(f"Error RMS  diferencias finitas vs Fourier : {np.sqrt(mse(u_df, u_ff)):.2e}")
    reportar(f"Error RMS  Fourier vs filtro gaussiano    : {np.sqrt(mse(u_ff, u_gs)):.2e}")

    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    mostrar(ax[0], luna, "Original (t = 0)")
    mostrar(ax[1], u_df, f"Diferencias finitas\n({pasos} pasos)")
    mostrar(ax[2], u_ff, "Solución por Fourier\n(un solo paso)")
    mostrar(ax[3], u_gs, f"Filtro gaussiano σ = {sigma:.2f}")
    guardar(fig, "acto1_tres_metodos.png")

    # Evolución temporal
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    for a, ti in zip(ax, [0, 1, 4, 16, 64]):
        mostrar(a, calor_fourier(luna, k, ti), f"t = {ti}")
    guardar(fig, "acto1_evolucion.png")

    # El calor como filtro pasa-bajas: espectro y función de transferencia 1D
    w = np.linspace(-np.pi, np.pi, 400)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for ti in [0.5, 2, 8]:
        ax[0].plot(w, np.exp(-k * w**2 * ti), label=f"t = {ti}")
    ax[0].set_title("Función de transferencia  H(ω) = exp(-k ω² t)")
    ax[0].set_xlabel("frecuencia ω (rad/píxel)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    espectro = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(luna))))
    ax[1].imshow(espectro, cmap="magma")
    ax[1].set_title("log|F{imagen}|  (centro = frecuencias bajas)")
    ax[1].axis("off")
    guardar(fig, "acto1_pasabajas.png")

    # Aplicación: eliminar ruido
    ruido_sigma = 0.10
    luna_ruido = np.clip(luna + rng.normal(0, ruido_sigma, luna.shape), 0, 1)
    tiempos = np.linspace(0, 6, 61)
    psnrs = [psnr(calor_fourier(luna_ruido, k, ti), luna) for ti in tiempos]
    t_opt = tiempos[int(np.argmax(psnrs))]
    reportar(f"Ruido gaussiano σ = {ruido_sigma}: PSNR ruidosa = {psnr(luna_ruido, luna):.2f} dB")
    reportar(f"Tiempo óptimo de difusión t* = {t_opt:.2f} -> PSNR = {max(psnrs):.2f} dB")
    reportar(f"Con t = 6 (demasiado tiempo) -> PSNR = {psnrs[-1]:.2f} dB")

    fig = plt.figure(figsize=(15, 4))
    a1 = fig.add_subplot(1, 4, 1); mostrar(a1, luna_ruido, "Con ruido")
    a2 = fig.add_subplot(1, 4, 2); mostrar(a2, calor_fourier(luna_ruido, k, t_opt), f"Difusión t* = {t_opt:.1f}")
    a3 = fig.add_subplot(1, 4, 3); mostrar(a3, calor_fourier(luna_ruido, k, 6), "Difusión t = 6 (de más)")
    a4 = fig.add_subplot(1, 4, 4)
    a4.plot(tiempos, psnrs, color="tab:red")
    a4.axvline(t_opt, ls="--", color="gray")
    a4.set_xlabel("tiempo de difusión t"); a4.set_ylabel("PSNR (dB)")
    a4.set_title("Calidad vs. tiempo"); a4.grid(alpha=0.3)
    guardar(fig, "acto1_ruido.png")


    # ACTO 2: fronteras, estabilidad y Gibbs
    reportar("")
    reportar("=" * 70)
    reportar("ACTO 2: condiciones de frontera, estabilidad y Gibbs")
    reportar("=" * 70)

    t2 = 60.0
    u_dir, _ = calor_diferencias(luna, k, t2, "dirichlet")
    u_neu, _ = calor_diferencias(luna, k, t2, "neumann")
    reportar(f"t = {t2}: brillo medio original = {luna.mean():.4f}")
    reportar(f"  Dirichlet (bordes a 0)   -> brillo medio = {u_dir.mean():.4f} (el calor se escapa)")
    reportar(f"  Neumann  (bordes aislados) -> brillo medio = {u_neu.mean():.4f} (se conserva)")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    mostrar(ax[0], luna, "Original")
    mostrar(ax[1], u_dir, "Dirichlet: bordes a temperatura 0")
    mostrar(ax[2], u_neu, "Neumann: bordes aislados")
    guardar(fig, "acto2_fronteras.png")

    # Estabilidad: r = 0.24 (estable) vs r = 0.26 (inestable)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for a, r in zip(ax, [0.24, 0.26]):
        u = luna.copy()
        for _ in range(150):
            u = u + r * laplaciano(u, "periodica")
        reportar(f"Estabilidad: r = {r} tras 150 pasos -> max|u| = {np.abs(u).max():.3e}")
        estado = "estable" if r <= 0.25 else "INESTABLE"
        a.imshow(np.clip(u, 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"r = k·dt/dx² = {r}  ({estado})", fontsize=10)
        a.axis("off")
    guardar(fig, "acto2_estabilidad.png")

    # Gibbs: filtro pasa-bajas ideal (corte brusco) vs gaussiano (calor)
    placa = imagen_placa()
    W2 = frecuencias(placa.shape)
    corte = 0.35 * np.pi
    ideal = np.real(np.fft.ifft2(np.fft.fft2(placa) * (W2 <= corte**2)))
    k_eq = 1.0 / corte**2
    gauss = calor_fourier(placa, 1.0, k_eq)
    fila = placa.shape[0] // 2
    sobre = ideal[fila].max() - 1.0
    reportar(f"Gibbs: sobrepaso del filtro ideal = {100 * sobre:.1f} % "
             f"(en 1D la teoría da ≈ 9 % del salto; en 2D con corte circular y bordes cercanos puede ser mayor); filtro del calor = {100 * max(gauss[fila].max() - 1, 0):.2f} %")

    fig, ax = plt.subplots(2, 2, figsize=(12, 5.5), gridspec_kw={"height_ratios": [1, 1.1]})
    mostrar(ax[0, 0], np.clip(ideal, 0, 1), "Pasa-bajas IDEAL (corte brusco)")
    mostrar(ax[0, 1], gauss, "Pasa-bajas del CALOR (gaussiano)")
    for a, perfil, nom in [(ax[1, 0], ideal, "ideal"), (ax[1, 1], gauss, "calor")]:
        a.plot(placa[fila], color="gray", lw=1, label="original")
        a.plot(perfil[fila], color="tab:blue", lw=1.3, label=nom)
        a.axhline(1, color="k", lw=0.5)
        a.set_xlim(0, placa.shape[1] - 1)
        a.set_title(f"Perfil de la fila {fila}")
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
    guardar(fig, "acto2_gibbs.png")


    # ACTO 3: el calor hacia atrás
    reportar("")
    reportar("=" * 70)
    reportar("ACTO 3: calor hacia atrás (desenfoque inverso)")
    reportar("=" * 70)

    T = 1.2
    H = np.exp(-k * frecuencias(placa.shape) * T)
    borrosa = np.real(np.fft.ifft2(np.fft.fft2(placa) * H))
    ruido = 1e-3
    borrosa_ruido = borrosa + rng.normal(0, ruido, placa.shape)

    reportar(f"Desenfoque con T = {T} (σ = {np.sqrt(2 * k * T):.2f} px). "
             f"Amplificación máxima 1/H = {1 / H.min():.2e}")


    rec_sin_ruido = calor_atras(borrosa, H)
    rec_con_ruido = calor_atras(borrosa_ruido, H)
    lams = np.logspace(-7, -1, 60)
    psnr_w = [psnr(np.clip(wiener(borrosa_ruido, H, l), 0, 1), placa) for l in lams]
    lam_opt = lams[int(np.argmax(psnr_w))]
    rec_wiener = np.clip(wiener(borrosa_ruido, H, lam_opt), 0, 1)

    reportar(f"PSNR borrosa (con ruido {ruido})            : {psnr(borrosa_ruido, placa):.2f} dB")
    reportar(f"PSNR inversión directa SIN ruido          : {psnr(np.clip(rec_sin_ruido, 0, 1), placa):.2f} dB")
    reportar(f"Inversión directa CON ruido: max|u| = {np.abs(rec_con_ruido).max():.2e}  (explota)")
    reportar(f"Wiener: λ óptimo = {lam_opt:.2e} -> PSNR = {max(psnr_w):.2f} dB")

    fig, ax = plt.subplots(2, 3, figsize=(15, 5.2))
    mostrar(ax[0, 0], placa, "Original")
    mostrar(ax[0, 1], borrosa_ruido, f"Desenfocada (calor T={T}) + ruido {ruido}")
    mostrar(ax[0, 2], np.clip(rec_sin_ruido, 0, 1), "Calor hacia atrás SIN ruido")
    ax[1, 0].imshow(rec_con_ruido, cmap="gray")
    ax[1, 0].set_title("Calor hacia atrás CON ruido (explota)", fontsize=10)
    ax[1, 0].axis("off")
    mostrar(ax[1, 1], rec_wiener, f"Inversión regularizada (λ={lam_opt:.1e})")
    ax[1, 2].semilogx(lams, psnr_w, color="tab:green")
    ax[1, 2].axvline(lam_opt, ls="--", color="gray")
    ax[1, 2].set_xlabel("λ (regularización)"); ax[1, 2].set_ylabel("PSNR (dB)")
    ax[1, 2].set_title("Elección de λ"); ax[1, 2].grid(alpha=0.3)
    guardar(fig, "acto3_calor_atras.png")

    # Por qué explota: ganancia de la inversión directa vs la regularizada (1D)
    w = np.linspace(0, np.pi, 400)
    h = np.exp(-k * w**2 * T)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(w, 1 / h, label="inversión directa  exp(+kω²T)")
    ax.semilogy(w, h / (h**2 + lam_opt), label=f"regularizada  H/(H²+λ)")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("frecuencia ω (rad/píxel)"); ax.set_ylabel("ganancia")
    ax.set_title("Amplificación de cada frecuencia al 'retroceder el tiempo'")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    guardar(fig, "acto3_ganancia.png")


    # EXTRA: Teorema de Parseval
    reportar("")
    reportar("=" * 70)
    reportar("EXTRA: teorema de Parseval")
    reportar("=" * 70)

    N = luna.size
    F = np.fft.fft2(luna)
    e_espacio = np.sum(luna**2)
    e_frec = np.sum(np.abs(F) ** 2) / N
    reportar(f"Energía en el espacio     Σ|u|²      = {e_espacio:.6f}")
    reportar(f"Energía en frecuencia (1/N)Σ|û|²     = {e_frec:.6f}")
    reportar(f"Diferencia relativa = {abs(e_espacio - e_frec) / e_espacio:.1e}")

    ts = np.linspace(0, 20, 81)
    W2l = frecuencias(luna.shape)
    Fd = F.copy()
    Fd[0, 0] = 0
    energia = [np.sum(np.abs(Fd * np.exp(-k * W2l * ti)) ** 2) / N for ti in ts]
    directa = np.sum((calor_fourier(luna, k, ts[-1]) - luna.mean()) ** 2)
    reportar(f"Energía de detalles t=0: {energia[0]:.2f}  ->  t=20: {energia[-1]:.2f} "
             f"({100 * energia[-1] / energia[0]:.1f} % de la inicial)")
    reportar(f"Comprobación t=20 calculando en el espacio: {directa:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, energia, color="tab:purple")
    ax.set_xlabel("tiempo t"); ax.set_ylabel("energía de los detalles")
    ax.set_title("Disipación de energía (calculada con Parseval)")
    ax.grid(alpha=0.3)
    guardar(fig, "extra_parseval.png")

    with open("resultados.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(resultados) + "\n")
    print(f"\nFiguras guardadas en ./{CARPETA}/  y resultados en resultados.txt")


# PARTE 6 · Ventana interactiva para la exposición
FONDO = "#1C1F26"
PANEL = "#262A33"
TEXTO = "#F1F3F6"
SUAVE = "#A0A7B4"
NARANJA = "#D9531E"
VERDE = "#1F8A84"
AMARILLO = "#F2A541"
LADO_MAX = 256


def cargar_foto(ruta):
    img = Image.open(ruta).convert("L")
    img.thumbnail((LADO_MAX, LADO_MAX), Image.LANCZOS)
    return np.asarray(img, dtype=float) / 255.0


class Deslizador(tk.Frame):

    def __init__(self, padre, texto, desde, hasta, inicial, al_cambiar, log=False, formato="{:.2f}"):
        super().__init__(padre, bg=PANEL)
        self.log, self.formato, self.al_cambiar = log, formato, al_cambiar
        a, b, v = (np.log10(desde), np.log10(hasta), np.log10(inicial)) if log else (desde, hasta, inicial)
        tk.Label(self, text=texto, bg=PANEL, fg=TEXTO, font=("Segoe UI", 11), width=22, anchor="w").pack(side="left")
        self.var = tk.DoubleVar(value=v)
        ttk.Scale(self, from_=a, to=b, variable=self.var, command=self._mover,
                  style="Calor.Horizontal.TScale", length=300).pack(side="left", padx=10)
        self.etiqueta = tk.Label(self, bg=PANEL, fg=AMARILLO, font=("Consolas", 12, "bold"), width=10, anchor="w")
        self.etiqueta.pack(side="left")
        self._mover()

    def valor(self):
        v = self.var.get()
        return 10 ** v if self.log else v

    def poner(self, v):
        self.var.set(np.log10(v) if self.log else v)
        self._mover(llamar=False)

    def _mover(self, *_, llamar=True):
        self.etiqueta.config(text=self.formato.format(self.valor()))
        if llamar and len(_) > 0:
            self.al_cambiar()


class Pestana(tk.Frame):

    titulos = ("", "", "")

    def __init__(self, padre, app):
        super().__init__(padre, bg=FONDO)
        self.app = app
        self._pendiente = None
        self.fig = Figure(figsize=(12, 4.2), facecolor=FONDO)
        self.ejes = [self.fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
        self.controles = tk.Frame(self, bg=PANEL, padx=18, pady=12)
        self.controles.pack(side="bottom", fill="x", padx=10, pady=10)
        self.metrica = tk.Label(self.controles, bg=PANEL, fg=AMARILLO, font=("Segoe UI", 13, "bold"),
                                justify="left", anchor="w")
        self.metrica.pack(side="bottom", fill="x", pady=(10, 0))
        self.izq = tk.Frame(self.controles, bg=PANEL)
        self.izq.pack(side="left", fill="y", anchor="n")
        self.explicacion = tk.Label(self.controles, bg=PANEL, fg=SUAVE, font=("Segoe UI", 11),
                                    justify="left", anchor="nw")
        self.explicacion.pack(side="left", fill="both", expand=True, padx=(30, 0))
        self.explicacion.bind("<Configure>", lambda e: self.explicacion.config(wraplength=max(e.width - 10, 200)))

        self.lienzo = FigureCanvasTkAgg(self.fig, master=self)
        self.lienzo.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 0))

    def deslizador(self, *args, **kw):
        d = Deslizador(self.izq, *args, al_cambiar=self.pedir_actualizar, **kw)
        d.pack(anchor="w", pady=3)
        return d

    def boton(self, texto, comando, padre=None):
        b = tk.Button(padre or self.izq, text=texto, command=comando, bg=VERDE, fg="white",
                      activebackground=NARANJA, activeforeground="white", relief="flat",
                      font=("Segoe UI", 10, "bold"), padx=12, pady=4, cursor="hand2")
        b.pack(anchor="w", pady=(6, 0))
        return b

    def pedir_actualizar(self):
        if self._pendiente:
            self.after_cancel(self._pendiente)
        self._pendiente = self.after(120, self.actualizar)

    def mostrar(self, i, img, titulo, color=TEXTO, rango=(0, 1), suavizar=True):
        ax = self.ejes[i]
        ax.clear()
        ax.imshow(img, cmap="gray", vmin=rango[0], vmax=rango[1],
                  interpolation="antialiased" if suavizar else "nearest")
        ax.set_title(titulo, color=color, fontsize=12, pad=8)
        ax.axis("off")

    def dibujar(self):
        self.lienzo.draw_idle()

    def actualizar(self):
        raise NotImplementedError


# Tema 1: condiciones de frontera
class PestanaFronteras(Pestana):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.t = self.deslizador("Tiempo de difusión  t", 0, 100, 40, formato="{:.0f}")
        self.inestable = tk.BooleanVar(value=False)
        tk.Checkbutton(self.izq, text="Usar r = 0.26 (romper la estabilidad, r > 1/4)",
                       variable=self.inestable, command=self.pedir_actualizar, bg=PANEL, fg=TEXTO,
                       selectcolor=FONDO, activebackground=PANEL, activeforeground=TEXTO,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))
        self.explicacion.config(text=(
            "Tema 1 — Ecuación del calor:  ∂u/∂t = k (∂²u/∂x² + ∂²u/∂y²)\n\n"
            "Dirichlet: los bordes se fijan a 0 (negro). El calor se escapa y la imagen se oscurece "
            "desde los bordes, como la barra con extremos a cero.\n\n"
            "Neumann: bordes aislados. No sale calor, así que el brillo promedio se conserva, "
            "como la barra con extremos aislados."))

    def actualizar(self):
        u0, t = self.app.imagen, self.t.valor()
        r = 0.26 if self.inestable.get() else 0.2
        u_d, pasos = calor_diferencias(u0, 1.0, t, "dirichlet", r=r)
        u_n, _ = calor_diferencias(u0, 1.0, t, "neumann", r=r)
        self.mostrar(0, u0, f"Original   (brillo {u0.mean():.3f})")
        self.mostrar(1, np.clip(u_d, 0, 1), f"Dirichlet: bordes a 0   (brillo {u_d.mean():.3f})", NARANJA)
        self.mostrar(2, np.clip(u_n, 0, 1), f"Neumann: aislado   (brillo {u_n.mean():.3f})", VERDE)
        if self.inestable.get():
            maximo = np.abs(u_n).max()
            estado = "explota" if maximo > 1 else "el error está creciendo, todavía no se nota"
            self.metrica.config(text=f"r = 0.26 > 1/4  →  {pasos} pasos, máximo |u| = {maximo:.1e}  ({estado})",
                                fg=NARANJA)
        else:
            self.metrica.config(text=f"Diferencias finitas: {pasos} pasos con r = 0.2  (estable, r ≤ 1/4)", fg=AMARILLO)
        self.dibujar()


# Tema 2: Fourier quita el ruido
class PestanaRuido(Pestana):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.ruido = self.deslizador("Cantidad de ruido  σ", 0.0, 0.3, 0.10)
        self.t = self.deslizador("Tiempo de difusión  t", 0.0, 10.0, 2.0)
        self.boton("Buscar el mejor tiempo t", self.mejor_t)
        self.explicacion.config(text=(
            "Tema 2 — Con la transformada de Fourier la EDP se vuelve una EDO por frecuencia:\n"
            "        U(ω,t) = F(ω) · e^(−kω²t)\n\n"
            "El factor e^(−kω²t) es un filtro pasa-bajas: apaga las frecuencias altas (ruido) y deja "
            "las bajas. Es lo mismo que desenfocar con una gaussiana de σ = √(2kt)."))

    def _ruidosa(self):
        return self.app.imagen + self.ruido.valor() * self.app.ruido_base

    def mejor_t(self):
        u0, ruidosa = self.app.imagen, self._ruidosa()
        ts = np.linspace(0, 10, 101)
        mejor = ts[int(np.argmax([psnr(calor_fourier(ruidosa, 1.0, t), u0) for t in ts]))]
        self.t.poner(mejor)
        self.actualizar()

    def actualizar(self):
        u0, t = self.app.imagen, self.t.valor()
        ruidosa = self._ruidosa()
        limpia = calor_fourier(ruidosa, 1.0, t)
        self.mostrar(0, u0, "Original")
        self.mostrar(1, np.clip(ruidosa, 0, 1), f"Con ruido   ({psnr(ruidosa, u0):.1f} dB)", NARANJA)
        self.mostrar(2, np.clip(limpia, 0, 1), f"Calor por Fourier, t = {t:.1f}   ({psnr(limpia, u0):.1f} dB)", VERDE)
        self.metrica.config(text=f"Calidad: {psnr(ruidosa, u0):.1f} dB  →  {psnr(limpia, u0):.1f} dB     "
                                 f"(desenfoque σ = √(2kt) = {np.sqrt(2 * t):.2f} px)")
        self.dibujar()


# Tema 3: calor hacia atrás
class PestanaReenfocar(Pestana):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.T = self.deslizador("Desenfoque  T", 0.2, 3.0, 1.2)
        self.ruido = self.deslizador("Ruido de la cámara", 1e-6, 1e-2, 1e-3, log=True, formato="{:.0e}")
        self.lam = self.deslizador("Regularización  λ", 1e-7, 1e-1, 3e-5, log=True, formato="{:.0e}")
        self.boton("Buscar el mejor λ", self.mejor_lam)
        self.explicacion.config(text=(
            "Tema 3 — Para retroceder el tiempo se divide por el filtro:  F = B · e^(+kω²T)\n"
            "Sin ruido funciona, pero con un ruido mínimo las frecuencias altas se multiplican por "
            "números enormes y la imagen explota (problema mal planteado).\n\n"
            "Filtro de Wiener:  F = H / (H² + λ) · B   invierte donde hay señal y no amplifica el ruido."))

    def _datos(self):
        u0 = self.app.imagen
        H = np.exp(-frecuencias(u0.shape) * self.T.valor())
        borrosa = np.real(np.fft.ifft2(np.fft.fft2(u0) * H)) + self.ruido.valor() * self.app.ruido_base
        return u0, H, borrosa

    def mejor_lam(self):
        u0, H, borrosa = self._datos()
        lams = np.logspace(-7, -1, 61)
        self.lam.poner(lams[int(np.argmax([psnr(np.clip(wiener(borrosa, H, l), 0, 1), u0) for l in lams]))])
        self.actualizar()

    def actualizar(self):
        u0, H, borrosa = self._datos()
        directa = calor_atras(borrosa, H)
        rec = np.clip(wiener(borrosa, H, self.lam.valor()), 0, 1)
        maximo = np.abs(directa).max()
        self.mostrar(0, np.clip(borrosa, 0, 1), f"Borrosa + ruido   ({psnr(borrosa, u0):.1f} dB)")
        if maximo > 2:
            lo, hi = np.percentile(directa, [2, 98])
            self.mostrar(1, directa, f"Calor hacia atrás: ¡explota! (máx {maximo:.0e})", NARANJA,
                         rango=(lo, hi), suavizar=False)
        else:
            self.mostrar(1, np.clip(directa, 0, 1), f"Calor hacia atrás   ({psnr(np.clip(directa, 0, 1), u0):.1f} dB)", NARANJA)
        self.mostrar(2, rec, f"Filtro de Wiener   ({psnr(rec, u0):.1f} dB)", VERDE)
        self.metrica.config(text=f"Amplificación:  inversión directa hasta {1 / H.min():.0e}   ·   "
                                 f"Wiener máximo {1 / (2 * np.sqrt(self.lam.valor())):.0f}")
        self.dibujar()


# Ventana principal
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("El calor como filtro — Matemáticas Especiales")
        self.configure(bg=FONDO)
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self._estilos()

        cab = tk.Frame(self, bg=FONDO, padx=20, pady=14)
        cab.pack(fill="x")
        titulos = tk.Frame(cab, bg=FONDO)
        titulos.pack(side="left")
        tk.Label(titulos, text="El calor como filtro", bg=FONDO, fg=TEXTO,
                 font=("Cambria", 24, "bold")).pack(anchor="w")
        tk.Label(titulos, text="ecuación del calor  ·  transformada de Fourier  ·  calor hacia atrás",
                 bg=FONDO, fg=SUAVE, font=("Segoe UI", 11)).pack(anchor="w")
        for texto, comando in (("Placa UCO", lambda: self.usar(imagen_placa(), "Placa UCO 2026")),
                               ("Luna", lambda: self.usar(imagen_luna(), "Luna")),
                               ("📂  Abrir foto", self.abrir)):
            tk.Button(cab, text=texto, command=comando, bg=NARANJA if "Abrir" in texto else PANEL,
                      fg="white", activebackground=VERDE, activeforeground="white", relief="flat",
                      font=("Segoe UI", 11, "bold"), padx=16, pady=6, cursor="hand2").pack(side="right", padx=4)

        self.pestanas = ttk.Notebook(self, style="Calor.TNotebook")
        self.pestanas.pack(fill="both", expand=True, padx=10)
        self.vistas = [PestanaFronteras(self.pestanas, self), PestanaRuido(self.pestanas, self),
                       PestanaReenfocar(self.pestanas, self)]
        for v, nombre in zip(self.vistas, ("  1 · Fronteras  ", "  2 · Quitar ruido (Fourier)  ",
                                           "  3 · Reenfocar (calor hacia atrás)  ")):
            self.pestanas.add(v, text=nombre)
        self.pestanas.bind("<<NotebookTabChanged>>", lambda e: self.vista_actual().actualizar())

        self.estado = tk.Label(self, bg=FONDO, fg=SUAVE, font=("Segoe UI", 10), anchor="w", padx=20, pady=6)
        self.estado.pack(fill="x")
        self.usar(imagen_luna(), "Luna")

    def _estilos(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Calor.TNotebook", background=FONDO, borderwidth=0)
        s.configure("Calor.TNotebook.Tab", background=PANEL, foreground=SUAVE,
                    font=("Segoe UI", 12, "bold"), padding=(14, 8), borderwidth=0)
        s.map("Calor.TNotebook.Tab", background=[("selected", NARANJA)], foreground=[("selected", "white")])
        s.configure("Calor.Horizontal.TScale", background=PANEL, troughcolor=FONDO,
                    bordercolor=PANEL, lightcolor=VERDE, darkcolor=VERDE)

    def vista_actual(self):
        return self.vistas[self.pestanas.index("current")]

    def usar(self, img, nombre):
        self.imagen = img
        self.ruido_base = np.random.default_rng(2026).standard_normal(img.shape)
        self.estado.config(text=f"Imagen: {nombre}   ({img.shape[1]} × {img.shape[0]} píxeles)")
        self.vista_actual().actualizar()

    def abrir(self):
        ruta = filedialog.askopenfilename(title="Escoge una foto", filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self.usar(cargar_foto(ruta), ruta.replace("\\", "/").split("/")[-1])
        except Exception as e:
            messagebox.showerror("No se pudo abrir", f"No se pudo leer la imagen:\n{e}")


# PARTE 7 · Arranque
if __name__ == "__main__":
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "ventana"
    if modo == "figuras":
        generar_figuras()
    else:
        App().mainloop()
