import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from sklearn.metrics import silhouette_score # Importa silhouette_score

# --- Configurazione Generale dell'App Streamlit ---
st.set_page_config(
    page_title="K-Means per Studenti Erasmus IULM",
    page_icon="🎓", # Icona di laurea
    layout="wide" # Utilizza l'intera larghezza dello schermo
)

# --- Intestazione e Descrizione dell'Applicazione ---
st.title("🎓 Analisi degli Interessi degli Studenti Erasmus con K-Means per la IULM")
st.markdown("""
Benvenuti! Questa app dimostra come l'algoritmo di **Clustering K-Means** può aiutare un ufficio internazionale dell'Università **IULM**
a identificare gruppi di studenti con interessi simili per gli scambi Erasmus, specialmente nel contesto del corso di laurea in **Intelligenza Artificiale per l'Impresa e la Società**.

Non ci sono etichette predefinite sulle preferenze degli studenti, ma dati su tre aspetti chiave che riflettono il piano di studi IULM:
* **Punteggio Interesse Tecnologia/Innovazione (AI, Data Science):** La propensione dello studente verso l'applicazione pratica dell'Intelligenza Artificiale, l'analisi dei dati, e le nuove tecnologie.
* **Punteggio Interesse Umanistico/Comunicativo (Etica, Società):** L'interesse dello studente per l'impatto etico e sociale della tecnologia, la comunicazione digitale, l'interazione uomo-macchina e gli studi culturali.
* **Punteggio Interesse Culturale/Sociale (Immersione Erasmus):** L'interesse dello studente per l'immersione nella cultura locale, l'apprendimento di nuove lingue, l'esplorazione del territorio e le attività sociali durante l'Erasmus.

K-Means ci aiuterà a scoprire automaticamente questi gruppi per proporre programmi e destinazioni più mirate!
""")

# --- 1. Generazione dei Dati Simulati (I Nostri "Studenti IULM") ---
st.header("Step 1: I Nostri Studenti Simulati (IULM-style)")
st.markdown("""
Iniziamo creando un dataset simulato che rappresenta 400 studenti.
Ogni studente è posizionato su un grafico (o spazio tridimensionale) in base ai suoi tre "Punteggi Interesse".
""")

np.random.seed(42) # Fissa il seed per risultati riproducibili (stessi studenti ogni volta)

# Definiamo 4 "tipi" di studenti Erasmus con interessi leggermente diversi, più in linea con IULM AI
# loc=[media_Tecnologia, media_Umanistico, media_Culturale], scale=[dev_std_Tecnologia, dev_std_Umanistico, dev_std_Culturale]
student_group1 = np.random.normal(loc=[8.5, 5.0, 6.0], scale=[1.0, 1.0, 1.0], size=(100, 3)) # "Tech Innovators": Alto interesse AI/Tech, medio umanistico, buon culturale.
student_group2 = np.random.normal(loc=[4.0, 8.0, 7.5], scale=[1.0, 1.0, 1.0], size=(120, 3)) # "Ethical & Cultural Explorers": Meno focus tech, molto interessati a etica/società e cultura.
student_group3 = np.random.normal(loc=[7.0, 7.0, 3.0], scale=[1.0, 1.0, 1.0], size=(90, 3))  # "Balanced Academics": Equilibrati tra tech e umanistico, meno interesse culturale.
student_group4 = np.random.normal(loc=[5.5, 5.0, 5.0], scale=[1.2, 1.2, 1.2], size=(80, 3))  # "All-Rounders": Interessati a tutti gli aspetti in modo bilanciato.

# Uniamo tutti i gruppi in un unico grande dataset
data = np.vstack((student_group1, student_group2, student_group3, student_group4))

# Assicuriamoci che i punteggi rimangano nel range logico di 0-10
data = np.clip(data, 0, 10)

feature_names = ["Punteggio Interesse Tecnologia/Innovazione", "Punteggio Interesse Umanistico/Comunicativo", "Punteggio Interesse Culturale/Sociale"]

st.info(f"Dataset generato: {data.shape[0]} profili studenti con {data.shape[1]} variabili per ognuno.")

# Visualizzazione iniziale dei dati (prima del clustering)
# Per semplicità, visualizziamo solo le prime due dimensioni per il 2D plot
fig_initial, ax_initial = plt.subplots(figsize=(10, 7))
ax_initial.scatter(data[:, 0], data[:, 1], s=50, alpha=0.7, color='#1f77b4', edgecolors='w', linewidth=0.5)
ax_initial.set_title("Distribuzione Iniziale dei Profili Studenti Erasmus (Tech vs Umanistico)", fontsize=16)
ax_initial.set_xlabel(feature_names[0], fontsize=12)
ax_initial.set_ylabel(feature_names[1], fontsize=12)
ax_initial.grid(True, linestyle='--', alpha=0.6)
ax_initial.set_xlim(0, 10)
ax_initial.set_ylim(0, 10)
st.pyplot(fig_initial)
st.markdown("I punti blu rappresentano i nostri studenti. Il K-Means ora li raggrupperà in base alla loro vicinanza.")


# --- 2. Funzioni Core del K-Means ---
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def initialize_centroids_kmeans_plusplus(data, k):
    n_samples = data.shape[0]
    centroids = np.empty((k, data.shape[1]))
    centroids[0] = data[np.random.randint(n_samples)]
    for i in range(1, k):
        distances = np.array([
            min([euclidean_distance(point, c) for c in centroids[:i]]) for point in data
        ])
        probabilities = distances ** 2 / np.sum(distances ** 2)
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = data[next_idx]
    return centroids

def assign_to_clusters(data, centroids):
    return np.array([
        np.argmin([euclidean_distance(p, c) for c in centroids]) for p in data
    ])

def update_centroids(data, assignments, k):
    new_centroids = np.array([
        data[assignments == i].mean(axis=0) if np.any(assignments == i) else np.zeros(data.shape[1])
        for i in range(k)
    ])
    return new_centroids

def calculate_inertia(data, assignments, centroids):
    inertia = sum(
        np.sum((data[assignments == i] - centroid) ** 2)
        for i, centroid in enumerate(centroids)
        if np.any(assignments == i)
    )
    return inertia


# --- 3. Funzione di Visualizzazione per Streamlit (Adattata per 3 dimensioni ma ne visualizza 2) ---
def plot_clusters(data, assignments, centroids, colors, markers, iteration, inertia, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.set_title(f"Iterazione {iteration} - Inerzia: {inertia:.2f} {title_suffix}", fontsize=16)
    ax.set_xlabel(feature_names[0], fontsize=12) # Ora usa il nuovo nome
    ax.set_ylabel(feature_names[1], fontsize=12) # E il secondo nuovo nome
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(len(centroids)):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], # Visualizza solo le prime 2 dimensioni
                       color=colors[i % len(colors)],
                       alpha=0.6, s=50, edgecolors='w', linewidth=0.5,
                       label=f'Gruppo {i+1} ({len(cluster_points)} studenti)')
            
            ax.scatter(centroids[i, 0], centroids[i, 1], # Visualizza solo le prime 2 dimensioni del centroide
                       marker=markers[i % len(markers)], s=400,
                       color=colors[i % len(colors)], edgecolor='black', linewidth=2,
                       zorder=10)
            
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    st.pyplot(fig)


# --- 4. La Funzione K-Means Completa con Animazione Interattiva ---
def kmeans_animated_advanced(data, k, max_iterations=30):
    st.subheader(f"Step 2: L'Algoritmo K-Means in Azione con K={k}")
    st.markdown("Guarda come i punti cambiano gruppo e i centri (i grandi marcatori) si spostano ad ogni iterazione!")
    
    centroids = initialize_centroids_kmeans_plusplus(data, k)
    
    colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#00CED1', '#FF4500', '#7FFF00']  
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', '+']  

    inertia_history = []
    old_centroids = None

    chart_placeholder = st.empty()
    status_message = st.empty()

    for iteration in range(max_iterations):
        assignments = assign_to_clusters(data, centroids)
        current_inertia = calculate_inertia(data, assignments, centroids)
        inertia_history.append(current_inertia)
        
        new_centroids = update_centroids(data, assignments, k)

        if old_centroids is not None and np.allclose(new_centroids, old_centroids, atol=1e-6):
            status_message.success(f"✅ Convergenza raggiunta all'iterazione {iteration + 1}! I gruppi sono stabili.")
            break
            
        old_centroids = new_centroids.copy()
        centroids = new_centroids

        with chart_placeholder:
            plot_clusters(data, assignments, centroids, colors, markers, iteration + 1, current_inertia)
        status_message.info(f"Iterazione {iteration + 1} completata. Inerzia attuale: {current_inertia:.2f}...")
        time.sleep(0.5)

    with chart_placeholder:
        plot_clusters(data, assignments, centroids, colors, markers, iteration + 1, current_inertia, " (Finale)")
    status_message.success(f"Clustering K-Means completato dopo {iteration + 1} iterazioni!")
    
    return assignments, centroids, inertia_history

# --- Sezione per l'Elbow Method e Silhouette ---
st.header("Step 2 (A): Determinare il numero ottimale di cluster (K)")
st.markdown("""
Per scegliere il numero ottimale di cluster (`K`), utilizziamo due metriche complementari:
* **Elbow Method (Inerzia - WCSS):** Valuta la compattezza interna dei cluster. Si cerca un "gomito" nel grafico, dove la diminuzione dell'inerzia rallenta drasticamente.
* **Coefficiente di Silhouette:** Misura quanto un punto è simile al proprio cluster (coesione) rispetto a quanto è dissimile dal cluster più vicino (separazione). I valori vanno da -1 a +1; un punteggio più alto (vicino a +1) indica cluster ben definiti e separati.
""")

run_metrics = st.checkbox("Esegui l'analisi delle metriche per suggerire K (potrebbe richiedere qualche secondo)")
if run_metrics:
    st.info("Calcolo dell'inerzia e del Coefficiente di Silhouette per diversi valori di K...")
    max_k_for_metrics = 10
    inertia_for_k = []
    silhouette_for_k = [] # Lista per i punteggi di Silhouette
    
    # Placeholder per visualizzare l'avanzamento
    progress_bar = st.progress(0)
    status_text = st.empty()

    for k_val in range(1, max_k_for_metrics + 1):
        if k_val == 1:
            # Per K=1, il centroide è la media di tutti i dati. L'inerzia è la somma delle distanze al quadrato da questa media.
            centroid_k1 = np.mean(data, axis=0)
            inertia_k = np.sum(np.sum((data - centroid_k1) ** 2, axis=1))
            # La Silhouette non è definita per K=1, la gestiamo per il plot
            silhouette_k = 0.0 # Useremo 0 o np.nan per non influenzare il plot da K=2
        else:
            # Esegui K-Means più volte e prendi il risultato migliore per inerzia e assegnazioni (per robustezza)
            best_inertia_k = float('inf')
            best_assignments_k = None # Per salvare le migliori assegnazioni per la Silhouette
            for _ in range(5): # Ripeti 5 volte per avere un risultato più stabile
                temp_centroids = initialize_centroids_kmeans_plusplus(data, k_val)
                for _ in range(10): # Poche iterazioni per convergere per l'elbow/silhouette
                    temp_assignments = assign_to_clusters(data, temp_centroids)
                    new_temp_centroids = update_centroids(data, temp_assignments, k_val)
                    if np.allclose(new_temp_centroids, temp_centroids, atol=1e-6):
                        break
                    temp_centroids = new_temp_centroids
            
                current_inertia = calculate_inertia(data, temp_assignments, temp_centroids)
                if current_inertia < best_inertia_k:
                    best_inertia_k = current_inertia
                    best_assignments_k = temp_assignments # Salva le assegnazioni migliori
            
            inertia_k = best_inertia_k
            
            # Calcola Silhouette solo se ci sono almeno 2 cluster e più di 1 punto nel cluster (per evitare errori)
            if k_val > 1 and len(np.unique(best_assignments_k)) > 1:
                try:
                    silhouette_k = silhouette_score(data, best_assignments_k)
                except ValueError:
                    # Questo può accadere se un cluster è vuoto o ha un solo punto, gestiamolo
                    silhouette_k = 0.0 # Assegna un valore neutro in caso di errore
            else:
                silhouette_k = 0.0 # Se K=1 o problemi con cluster, Silhouette non è definita

        inertia_for_k.append(inertia_k)
        silhouette_for_k.append(silhouette_k) # Aggiungi alla lista
        
        progress_bar.progress(k_val / max_k_for_metrics)
        status_text.text(f"Calcolato K={k_val}...")

    status_text.success("Calcolo delle metriche completato!")

    # Plot dell'Inerzia (Elbow Method)
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(range(1, max_k_for_metrics + 1), inertia_for_k, marker='o', linestyle='-', color='purple')
    ax_elbow.set_title("Elbow Method (Inerzia - WCSS)", fontsize=16)
    ax_elbow.set_xlabel("Numero di Cluster (K)", fontsize=12)
    ax_elbow.set_ylabel("Inerzia (WCSS)", fontsize=12)
    ax_elbow.grid(True, linestyle='--', alpha=0.6)
    ax_elbow.set_xticks(range(1, max_k_for_metrics + 1))
    st.pyplot(fig_elbow)
    st.markdown("Cerca il 'gomito' nel grafico dell'Inerzia, dove la diminuzione rallenta bruscamente. Questo è un buon `K`.")

    # Nuovo Plot per la Silhouette
    fig_silhouette, ax_silhouette = plt.subplots(figsize=(10, 6))
    # Il plot della Silhouette parte da K=2 perché non è definita per K=1
    ax_silhouette.plot(range(2, max_k_for_metrics + 1), silhouette_for_k[1:], marker='o', linestyle='-', color='blue')  
    ax_silhouette.set_title("Coefficiente di Silhouette Medio", fontsize=16)
    ax_silhouette.set_xlabel("Numero di Cluster (K)", fontsize=12)
    ax_silhouette.set_ylabel("Punteggio di Silhouette Medio", fontsize=12)
    ax_silhouette.grid(True, linestyle='--', alpha=0.6)
    ax_silhouette.set_xticks(range(2, max_k_for_metrics + 1)) # I tick partono da 2
    st.pyplot(fig_silhouette)
    st.markdown("Cerca il picco più alto nel grafico della Silhouette. Un punteggio più vicino a +1 indica un clustering migliore.")
    st.markdown("---")

# Slider per selezionare il numero di cluster (K)
k_clusters_to_find = st.slider(
    "Quanti gruppi di studenti (K) vuoi trovare? (Usa i grafici sopra per un suggerimento!)",
    min_value=2,
    max_value=6,
    value=4, # Valore di default
    step=1
)

if st.button("Avvia Analisi K-Means"):
    final_assignments, final_centroids, _ = kmeans_animated_advanced(data, k_clusters_to_find) # Non catturiamo più inertia_trace se non lo usiamo

    # --- 6. Output dei Risultati Finali ---
    st.header("Step 3: I Risultati del Clustering Finale")
    st.success("Analisi K-Means completata! Ecco i profili e le dimensioni dei gruppi identificati.")

    st.subheader("📌 Profili dei Centroidi (Il 'Centro' di Ogni Gruppo):")
    st.markdown("Ogni centroide rappresenta il profilo medio degli studenti all'interno di quel gruppo:")
    
    cluster_descriptions = []
    for i, centroid in enumerate(final_centroids):
        desc = f"**Gruppo {i+1}:** "
        
        tech_score = centroid[0]
        hum_score = centroid[1]
        cult_score = centroid[2]

        desc += f"*{feature_names[0]}* = **{tech_score:.2f}**, "
        desc += f"*{feature_names[1]}* = **{hum_score:.2f}**, "
        desc += f"*{feature_names[2]}* = **{cult_score:.2f}**"
        
        # Aggiungi un'interpretazione qualitativa basata sulle tue simulazioni
        if tech_score > 7.0 and hum_score < 5.0 and cult_score > 5.0: # Tipo "Tech Innovators"
            desc += " - Questo gruppo è prevalentemente **orientato alla Tecnologia/Innovazione (AI, Data Science)**, con buon interesse culturale."
        elif hum_score > 7.0 and cult_score > 7.0 and tech_score < 5.0: # Tipo "Ethical & Cultural Explorers"
            desc += " - Questo gruppo è focalizzato sugli **Aspetti Umanistici/Comunicativi (Etica, Società)** e un forte **interesse Culturale/Sociale**."
        elif tech_score > 6.0 and hum_score > 6.0 and cult_score < 5.0: # Tipo "Balanced Academics"
            desc += " - Questo gruppo ha un **interesse bilanciato tra Tecnologia e Umanistica**, ma meno focus culturale."
        elif tech_score < 6.0 and hum_score < 6.0 and cult_score < 6.0: # Tipo "All-Rounders" ma con valori più bassi
              desc += " - Questo gruppo ha un **profilo di interessi bilanciato o meno marcato** su tutte le dimensioni."
        else:
            desc += " - Questo gruppo ha un **profilo di interessi misto o bilanciato**." # Catch-all
        
        cluster_descriptions.append(desc)
        st.write(desc)
    st.markdown("---")

    st.subheader("👥 Distribuzione degli Studenti nei Gruppi:")
    st.markdown("Ecco quanti studenti sono stati assegnati a ciascun gruppo:")
    unique_assignments, counts = np.unique(final_assignments, return_counts=True)
    for i, count in zip(unique_assignments, counts):
        st.write(f"**Gruppo {i+1}:** **{count}** studenti")
    st.markdown("---")

    # --- Calcolo e Visualizzazione del Coefficiente di Silhouette Finale ---
    st.subheader("📊 Qualità del Clustering: Coefficiente di Silhouette Finale")
    # Calcola la Silhouette solo se il numero di cluster è > 1 e ci sono abbastanza punti
    if k_clusters_to_find > 1 and len(np.unique(final_assignments)) > 1:
        try:
            final_silhouette_score = silhouette_score(data, final_assignments)
            st.write(f"Il **Coefficiente di Silhouette medio finale** per K={k_clusters_to_find} è: **{final_silhouette_score:.2f}**")
            st.markdown("""
            * Un punteggio vicino a **+1** indica che i cluster sono **ben separati e distinti**.
            * Un punteggio vicino a **0** indica cluster sovrapposti o ambigui.
            * Un punteggio vicino a **-1** suggerisce che i punti potrebbero essere stati assegnati al cluster sbagliato.
            """)
        except ValueError:
            st.warning("Impossibile calcolare il Coefficiente di Silhouette per la configurazione attuale dei cluster (es. cluster vuoti o con un solo punto).")
    else:
        st.info("Il Coefficiente di Silhouette non è applicabile per K=1 o se non sono stati formati cluster validi.")
    st.markdown("---")

    # --- 7. Simulazione e Suggerimento Destinazioni Erasmus ---
    st.header("Step 4: Suggerimento Destinazioni Erasmus Personalizzate")
    st.markdown("""
    Immaginiamo di avere alcune università partner Erasmus, ognuna con un profilo di "interessi" specifico,
    simili a quelli dei nostri studenti. Possiamo ora abbinare i gruppi di studenti alle destinazioni più affini.
    """)

    # New section explaining "Distanza dal profilo del gruppo"
    st.subheader("Come avviene l'abbinamento: La 'Distanza dal Profilo'")
    st.markdown("""
    Per suggerire le destinazioni più adatte a ciascun gruppo, calcoliamo la **distanza euclidea**
    tra il **profilo di interessi medio del gruppo (il centroide)** e il **profilo di interessi di ogni destinazione Erasmus**.
    
    * Una **distanza minore** significa che i profili sono più **simili**: il gruppo di studenti si troverebbe bene in quella destinazione.
    * Una **distanza maggiore** indica profili più **dissimili**.
    
    Questo ci permette di trovare le destinazioni che meglio si allineano con gli interessi predominanti di ogni gruppo di studenti.
    """)
    st.markdown("---") # Horizontal line for separation

    # Definiamo alcune destinazioni simulate con profili IULM-friendly
    # [Tech/Innovazione, Umanistico/Comunicativo, Culturale/Sociale]
    erasmus_destinations = {
        "Berlino (TU Berlin)": [9.0, 4.0, 7.0],  # Alto Tech, medio culturale, basso umanistico
        "Barcellona (UPF)": [7.0, 7.5, 8.5], # Equilibrato tech/um, alto culturale
        "Parigi (Sorbonne)": [4.0, 9.0, 9.0], # Alto umanistico/culturale, basso tech
        "Dublino (Trinity College)": [8.0, 6.0, 6.0], # Buon tech, buon umanistico, buon culturale
        "Amsterdam (UvA)": [6.5, 7.0, 8.0], # Equilibrato, molto culturale
        "Monaco (TUM)": [9.5, 3.5, 5.0], # Molto Tech, meno umanistico/culturale
    }

    st.subheader("🏫 Profili delle Destinazioni Erasmus Simulate:")
    for dest, profile in erasmus_destinations.items():
        st.write(f"**{dest}:** *{feature_names[0]}* = **{profile[0]:.1f}**, *{feature_names[1]}* = **{profile[1]:.1f}**, *{feature_names[2]}* = **{profile[2]:.1f}**")

    st.subheader("🎯 Abbinamento Studenti-Destinazioni:")
    st.markdown("Per ogni gruppo di studenti, suggeriamo le destinazioni più vicine al loro profilo medio:")

    for i, centroid in enumerate(final_centroids):
        st.write(f"**Suggerimenti per il Gruppo {i+1} (con {np.sum(final_assignments == i)} studenti):**")
        
        distances_to_destinations = {}
        for dest_name, dest_profile in erasmus_destinations.items():
            dist = euclidean_distance(centroid, np.array(dest_profile))
            distances_to_destinations[dest_name] = dist
        
        # Ordina le destinazioni per distanza (le più vicine per prime)
        sorted_destinations = sorted(distances_to_destinations.items(), key=lambda item: item[1])
        
        # Mostra le 3 destinazioni più vicine
        for j, (dest_name, dist) in enumerate(sorted_destinations[:3]):
            st.write(f"- **{dest_name}** (Distanza dal profilo del gruppo: **{dist:.2f}** - *minore è, migliore è l'abbinamento*)")
        st.markdown("") # Spazio tra i gruppi

    st.markdown("---")
    st.info("""
    **Conclusioni e Prossimi Passi per l'Ufficio Internazionale IULM:**
    L'analisi K-Means ha fornito una chiara suddivisione degli studenti in base ai loro interessi.
    Queste informazioni sono preziose per:
    * **Consigliare Destinazioni Mirate:** L'ufficio può ora abbinare in modo più efficace gli studenti a università partner che offrono un ambiente accademico e culturale in linea con i loro profili (es. un "Tech Innovator" a Berlino o Dublino, un "Ethical & Cultural Explorer" a Parigi o Barcellona).
    * **Personalizzare la Comunicazione:** Creare campagne informative e materiali promozionali specifici per ogni gruppo, evidenziando gli aspetti più pertinenti della vita universitaria e cittadina all'estero.
    * **Sviluppare Nuovi Accordi:** Identificare se ci sono tipi di studenti (cluster) per i quali mancano destinazioni ideali, orientando la ricerca di nuove partnership internazionali.
    * **Organizzare Eventi di Preparazione:** Sessioni informative pre-partenza possono essere strutturate per affrontare le domande e le aspettative specifiche di ciascun gruppo di studenti.
    """)