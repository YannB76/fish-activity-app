# 🎣 Fish Activity App – Le Havre

Application **Streamlit** pour estimer l’activité du **bar (dicentrarchus labrax)** au Havre sur les 3 prochains jours, en fonction de plusieurs paramètres environnementaux :

- 🌡️ Température de l’air et de l’eau  
- 🌬️ Vent (vitesse, direction)  
- 🌊 Houle (hauteur, période)  
- 🌥️ Nébulosité  
- 🌙 Phase de lune  
- 🌊 Marées (via WorldTides)  
- 📈 Pression atmosphérique et tendance  

---

## 🚀 Déploiement avec Streamlit Cloud

1. **Créer un compte** sur [Streamlit Cloud](https://streamlit.io/cloud)  
2. **Connecter ton GitHub** et choisir ce repo (`fish-activity-app`)  
3. **Configurer l’app** :
   - *Main file path* : `streamlit_app.py`  
   - *Python version* : 3.11 ou plus récent  
   - *Requirements file* : `requirements.txt`  

4. **Définir les secrets** : dans *Settings → Secrets* ajoute par exemple :

```toml
# Clé API WorldTides (obligatoire pour marées)
WORLDTIDES_API_KEY = "ta_cle_api_worldtides"

# (Optionnel) API locale pour la SST
# LOCAL_SST_URL = "https://ton-api-exemple.com/api/temperature/eau"
