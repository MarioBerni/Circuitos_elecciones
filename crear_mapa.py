# crear_mapa.py
import folium
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.colors as mcolors
from shapely.geometry import MultiPoint

# Datos estructurados basados en la imagen proporcionada
puntos = [
    {"titulo": "ESCUELA Nº 231", "direccion": "JUAN RAMÓN GÓMEZ 3082 ESQ. MARIANO MORENO", "coordenadas": (-34.881570862895465, -56.15645957511949), "circuitos": [260, 261, 262, 263]},
    {"titulo": "COLEGIO SANTA MARÍA", "direccion": "AV. 8 DE OCTUBRE 2966 ESQ. JAIME CIBILS", "coordenadas": (-34.884938758764946, -56.15654364628337), "circuitos": [264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274]},
    {"titulo": "ESCUELA Nº 19 - Nº 76", "direccion": "AV. 8 DE OCTUBRE 3515 ESQ. FELIPE SANGUINETTI", "coordenadas": (-34.877904935354714, -56.146033849986274), "circuitos": [634, 635, 636, 637, 638, 639, 640, 641, 642]},
    {"titulo": "ESCUELA Nº 20", "direccion": "AV. 8 DE OCTUBRE 3545 ESQ. FELIPE SANGUINETTI", "coordenadas": (-34.87799295172138, -56.14594801929908), "circuitos": [643, 644, 645, 646, 647, 648]},
    {"titulo": "POLICLÍNICA YUCATÁN", "direccion": "TOMÁS CLARAMUNT 3749 ESQ. GOBERNADOR VIANA", "coordenadas": (-34.86673771912881, -56.1488960904638), "circuitos": [649, 650, 651, 652]},
    {"titulo": "ESCUELA Nº 89", "direccion": "ALGARROBO 3719 ESQ. ING. JOSÉ SERRATO", "coordenadas": (-34.86527661913369, -56.148688775120156), "circuitos": [653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663]},
    {"titulo": "LICEO Nº 64", "direccion": "PARMA 3026 ESQ. AV. DÁMASO ANTONIO LARRAÑAGA", "coordenadas": (-34.867943840434314, -56.15210884628427), "circuitos": [664, 665, 666, 667]},
    {"titulo": "ANTEL ARENA - PUERTA S", "direccion": "AV. JOSÉ PEDRO VARELA S/N ESQ. AV. DÁMASO ANTONIO LARRAÑAGA", "coordenadas": (-34.8640346198508, -56.15229657697155), "circuitos": [668, 669, 670, 671, 672]},
    {"titulo": "CENTRO EDUCATIVO PRIMARIO ADVENTISTA", "direccion": "ING. JOSÉ SERRATO 3630 ESQ. BRUSELAS", "coordenadas": (-34.85512469145509, -56.15877780395654), "circuitos": [673, 674, 675, 676]},
    {"titulo": "COLEGIO REGINA MARTYRUM - PRIMARIA", "direccion": "ING. JOSÉ SERRATO 3536 ESQ. CORUMBÉ", "coordenadas": (-34.85604790135723, -56.156657976971864), "circuitos": [676, 677, 678]},
    {"titulo": "COLEGIO DOMINGO SAVIO", "direccion": "GERÓNIMO PICCIOLI 3272 ESQ. OSVALDO CRUZ", "coordenadas": (-34.85401729744644, -56.14013773464384), "circuitos": [1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045]},
    {"titulo": "ESCUELA Nº 117 - Nº 165", "direccion": "LABARDEN 4168 ESQ. CNO. CORRALES", "coordenadas": (-34.86186461256087, -56.140058403956225), "circuitos": [1046, 1047, 1048, 1049, 1050, 1051, 1052]},
    {"titulo": "LICEO Nº 76", "direccion": "SALGUERO 4345 ESQ. RAMÓN CASTRIZ", "coordenadas": (-34.86830141965752, -56.13065890395595), "circuitos": [1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060]},
    {"titulo": "LICEO Nº 19", "direccion": "20 DE FEBRERO 2510 ESQ. JOSÉ ANTONIO CABRERA", "coordenadas": (-34.86947081711175, -56.1307621058072), "circuitos": [1061, 1062, 1063, 1064, 1065, 1066]},
    {"titulo": "C.C.Z. Nº 9", "direccion": "AV. 8 DE OCTUBRE 4700 ESQ. MARCOS SASTRE", "coordenadas": (-34.85839499845363, -56.13368291930011), "circuitos": [1119, 1120]},
    {"titulo": "LICEO Nº 26", "direccion": "DR. JOAQUÍN REQUENA 3010 ESQ. ANTONIO MACHADO", "coordenadas": (-34.87110592801234, -56.17374008861227), "circuitos": [1397,  1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409]},
    {"titulo": "ESCUELA Nº 109 - Nº 156", "direccion": "AV. DR. LUIS A. DE HERRERA 3406 ESQ. MARNE", "coordenadas": (-34.86857065738785, -56.16886591787739), "circuitos": [1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418]},
    {"titulo": "ESCUELA Nº 40", "direccion": "PEDERNAL 1928 ESQ. PORONGOS", "coordenadas": (-34.87507993994863, -56.176301648135144), "circuitos": [1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426]},
    {"titulo": "ESCUELA Nº 11", "direccion": "DR. GUSTAVO GALLINAL 2130 ESQ. CUFRÉ", "coordenadas": (-34.8729937404566, -56.17123033464299), "circuitos": [1427, 1428, 1429]},
    {"titulo": "INSTITUTO PALLOTTI", "direccion": "EMILIO RAÑA S/N ESQ. JOAQUÍN CAMPANA", "coordenadas": (-34.87602593254925, -56.15588561929911), "circuitos": [1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442]},
    {"titulo": "INSTITUTO PALLOTTI", "direccion": "AV. DR. LUIS A. DE HERRERA 2882 ESQ. EMILIO RAÑA", "coordenadas": (-34.87691765003982, -56.15699045568918), "circuitos": [1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454]},
    {"titulo": "COLEGIO NUEVA CULTURA - INICIAL", "direccion": "AV. DR. LUIS A. DE HERRERA 2762 ESQ. MONTE CASEROS", "coordenadas": (-34.879247854151416, -56.15520643227576), "circuitos": [1455, 1456, 1457, 1458]},
    {"titulo": "LICEO ADVENTISTA DE MONTEVIDEO", "direccion": "AV. DR. LUIS A. DE HERRERA 2826 ESQ. EMILIO RAÑA", "coordenadas": (-34.876930603296, -56.15696699969745), "circuitos": [1459, 1460, 1461, 1462]},
    {"titulo": "UNIVERSIDAD DE LA EMPRESA", "direccion": "THOMPSON 3080 ESQ. AV. DR. LUIS A. DE HERRERA", "coordenadas": (-34.87586336314139, -56.15777887975017), "circuitos": [1463, 1464, 1465, 1466, 1467]},
    {"titulo": "PARROQUIA SAN ANTONINO", "direccion": "CARAGUATAY 2086 ESQ. CUFRÉ", "coordenadas": (-34.87673184821209, -56.17088030395562), "circuitos": [1468, 1469]},
    {"titulo": "ESCUELA Nº 21", "direccion": "VILARDEBÓ 1539 ESQ. AV. GRAL. SAN MARTÍN", "coordenadas": (-34.876280047535154, -56.18556755977615), "circuitos": [1470, 1471, 1472]},
    {"titulo": "ESCUELA DE EDUCACIÓN ARTÍSTICA Nº 310", "direccion": "AV. BURGUES 2735 ESQ. AV. GRAL. SAN MARTIN", "coordenadas": (-34.87529361943999, -56.185183132791536), "circuitos": [1473, 1474, 1475, 1476]},
    {"titulo": "ESCUELA Nº 68", "direccion": "CARABELAS 3279 ESQ. REGIMIENTO 9", "coordenadas": (-34.864197832526706, -56.181405388612575), "circuitos": [1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498]},
    {"titulo": "COLEGIO ATAHUALPA - PRIMARIA E INICIAL", "direccion": "CARMELO 1415 ESQ. DR. CARLOS VAZ FERREIRA", "coordenadas": (-34.86231122330693, -56.1916646039563), "circuitos": [1499, 1500, 1501]},
    {"titulo": "COLEGIO SAGRADO CORAZÓN", "direccion": "AV. GRAL. JOSÉ GARIBALDI 1682 ESQ. MARSELLA", "coordenadas": (-34.87466823919955, -56.183103432791526), "circuitos": [1502, 1503, 1504]},
    {"titulo": "U.T.U. BRAZO ORIENTAL", "direccion": "REGIMIENTO 9 1983 ESQ. DR. MANUEL LANDEIRA", "coordenadas": (-34.86619376887499, -56.17631650438428), "circuitos": [1505, 1506, 1507, 1508, 1509, 1510, 1511]},
    {"titulo": "ESCUELA Nº 90 DE 2DO. GRADO", "direccion": "AV. GRAL. FLORES 3013 ESQ. LORENZO FERNÁNDEZ", "coordenadas": (-34.87279394884114, -56.17675564813524), "circuitos": [1512, 1513, 1514, 1515]},
    {"titulo": "LICEO Nº 53", "direccion": "GUAVIYÚ S/N ESQ. REGIMIENTO 9", "coordenadas": (-34.86516622190964, -56.177612761627906), "circuitos": [1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524]},
    {"titulo": "ESCUELA Nº 90 DE 2DO. GRADO - ANEXO", "direccion": "ANTONIO MACHADO S/N ESQ. GRAL. FLORES", "coordenadas": (-34.87125614000635, -56.175392530940385), "circuitos": [1525, 1526]},
    {"titulo": "CAMBADU", "direccion": "AV. DR. LUIS A. DE HERRERA 4196 ESQ. CARLOS VAZ FERREIRA", "coordenadas": (-34.859114869241, -56.1899714258848), "circuitos": [1527, 1528, 1529, 1530]},
    {"titulo": "COLEGIO CLARA JACKSON DE HEBER", "direccion": "AV. DR. LUIS A. DE HERRERA 4142 ESQ. AV. BURGUES", "coordenadas": (-34.861011504318775, -56.18626555079144), "circuitos": [1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539]},
    {"titulo": "CENTRO EDUCACIONAL INTEGRAL", "direccion": "AV. DR. LUIS A. DE HERRERA 4092 ESQ. AV. BURGUES", "coordenadas": (-34.86106555094684, -56.18629531396111), "circuitos": [1540, 1541, 1542, 1543, 1544, 1545]},
    {"titulo": "ESCUELA Nº 136 - Nº 101", "direccion": "IBIROCAHY 3618 ESQ. ING. MANUEL RODRÍGUEZ CORREA", "coordenadas": (-34.85525293225015, -56.18510710395668), "circuitos": [1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558]},
    {"titulo": "COLEGIO MONSEÑOR JOSÉ BENITO LAMAS", "direccion": "IBIROCAHY 3785 ESQ. BV. JOSÉ BATLLE Y ORDÓÑEZ", "coordenadas": (-34.851706646209074, -56.183703219300405), "circuitos": [1559, 1560, 1561, 1562]},
    {"titulo": "COLEGIO JUAN MANUEL BLANES", "direccion": "AV. MILLÁN 4024 ESQ. COSTANERA FRANCISCO LAVALLEJA", "coordenadas": (-34.85216640264283, -56.19737320395675), "circuitos": [1563, 1564, 1565, 1566, 1567]},
    {"titulo": "ANGLO PRADO", "direccion": "AV. MILLÁN 3984 ESQ. RBLA. COSTANERA FRANCISCO LAVALLEJA", "coordenadas": (-34.852148793847924, -56.19737320395674), "circuitos": [1568, 1569]},
    {"titulo": "ESCUELA Nº 323", "direccion": "MATÍAS ÁLVAREZ S/N ESQ. PSJE. CABALLEROS ORIENTALES", "coordenadas": (-34.855006872797986, -56.195446506228656), "circuitos": [1570, 1571, 1572]},
    {"titulo": "LICEO Nº 18", "direccion": "AV. MILLÁN 3898 ESQ. AV. DR. LUIS A. DE HERRERA", "coordenadas": (-34.855787420031106, -56.19642407825556), "circuitos": [1573, 1574, 1575, 1576]},
    {"titulo": "U.T.U. INSTITUTO TECNOLÓGICO SUPERIOR", "direccion": "BV. JOSÉ BATLLE Y ORDÓÑEZ 3570 ESQ. AV. GRAL FLORES", "coordenadas": (-34.863803403766795, -56.16950604628437), "circuitos": [1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585]},
    {"titulo": "LICEO Nº 41", "direccion": "LEÓN PÉREZ 3800 ESQ. AV. GRAL. SAN MARTÍN", "coordenadas": (-34.85325052835129, -56.17219291751396), "circuitos": [1604, 1605, 1606, 1607, 1608]},
    {"titulo": "PLAZA DE DEPORTES Nº 4", "direccion": "LEÓN PÉREZ 3726 ESQ. FRANCISCO ROMERO", "coordenadas": (-34.85332586973235, -56.171658043365), "circuitos": [1609, 1610, 1611, 1612, 1613]},
    {"titulo": "COLEGIO DANIEL MCMURRAY", "direccion": "AV. GRAL. SAN MARTÍN 4012 ESQ. SANTA ANA", "coordenadas": (-34.852343436387926, -56.171638243365024), "circuitos": [1620, 1621, 1622]},
    {"titulo": "ESCUELA Nº 53 - Nº 87", "direccion": "JUAN ARTEAGA S/N ESQ. LEÓN PÉREZ", "coordenadas": (-34.85357337199852, -56.17062570103679), "circuitos": [1623, 1624, 1625, 1626,  1627, 1628, 1629, 1630, 1631]},
    {"titulo": "ESCUELA Nº 7 - Nº 37", "direccion": "JUAN ARTEAGA 4066 ESQ. LEÓN PÉREZ", "coordenadas": (-34.85301473946863, -56.17021694336503), "circuitos": [1632, 1633, 1634, 1635, 1636, 1637, 1638]},
    {"titulo": "CENTRO DE SALUD ANTONIO GIORDANO", "direccion": "AV. GRAL. SAN MARTÍN 3797 ESQ. RAFAEL HORTIGUERA", "coordenadas": (-34.85475771118329, -56.17331471930019), "circuitos": [1635]},
    {"titulo": "COLEGIO SAN PABLO", "direccion": "VENANCIO BENAVÍDEZ 3612 ESQ. 19 DE ABRIL", "coordenadas": (-34.86078153816523, -56.1956383039564), "circuitos": [1830, 1831, 1832, 1834, 1835, 1836, 1837]},
    {"titulo": "LICEO PROVIDENCIA PAPA FRANCISCO", "direccion": "ESTADOS UNIDOS 2497 ESQ. CNO. CIBILS", "coordenadas": (-34.87784085150969, -56.265987519299124), "circuitos": [2108, 2109]},
    {"titulo": "LICEO Nº 50", "direccion": "AV. GRAL. EDUARDO DA COSTA S/N ESQ. EX CNO. AL FRIGORÍFICO NACIONAL", "coordenadas": (-34.890493816191864, -56.26794679628506), "circuitos": [2110, 2111, 2112, 2113]}
]

# Puntos que se manejarán individualmente
puntos_individuales = [
    {"titulo": "LICEO PROVIDENCIA PAPA FRANCISCO", "direccion": "ESTADOS UNIDOS 2497 ESQ. CNO. CIBILS", "coordenadas": (-34.87784085150969, -56.265987519299124), "circuitos": [2108, 2109]},
    {"titulo": "LICEO Nº 50", "direccion": "AV. GRAL. EDUARDO DA COSTA S/N ESQ. EX CNO. AL FRIGORÍFICO NACIONAL", "coordenadas": (-34.890493816191864, -56.26794679628506), "circuitos": [2110, 2111, 2112, 2113]}
]

# Filtrar los puntos para excluir los individuales
puntos_restantes = [punto for punto in puntos if punto not in puntos_individuales]

# Convertir las coordenadas a un array de numpy para usar en el agrupamiento
coordenadas_restantes = np.array([punto["coordenadas"] for punto in puntos_restantes])

# Usar AgglomerativeClustering para agrupar los puntos restantes en 4 clusters
agg_clustering = AgglomerativeClustering(n_clusters=4).fit(coordenadas_restantes)
labels = agg_clustering.labels_

# Crear grupos basados en los labels del clustering
grupos = {i: [] for i in range(4)}
for punto, label in zip(puntos_restantes, labels):
    grupos[label].append(punto)

# Función para calcular la suma de circuitos en un grupo
def suma_circuitos(grupo):
    return sum(len(p["circuitos"]) for p in grupo)

# Crear el mapa centrado en Montevideo
m = folium.Map(location=[-34.8833, -56.1667], zoom_start=13)

# Definir colores para los grupos
colores = ['red', 'blue', 'green', 'purple']
colores_claros = [mcolors.to_hex(mcolors.to_rgba(c, alpha=0.2)) for c in colores]

# Estilo del tooltip
tooltip_style = """
<div style="font-family: Arial; font-size: 12pt; padding: 5px; border: 2px solid black; border-radius: 5px; background: white;">
    <strong>{titulo}</strong><br>
    Circuitos:<br>
    <ul>
    {circuitos}
    </ul>
    <footer style="font-size: 10pt; color: gray;">{direccion}</footer>
</div>
"""

# Función para calcular el centro de un grupo
def calcular_centro(grupo):
    coordenadas = np.array([punto["coordenadas"] for punto in grupo])
    return np.mean(coordenadas, axis=0)

# Añadir los marcadores al mapa con colores según el grupo
for i, grupo in grupos.items():
    # Delimitar el área del grupo con un polígono convexo
    poly_points = [punto["coordenadas"] for punto in grupo]
    
    if len(poly_points) >= 3:
        poly = MultiPoint(poly_points).convex_hull
        folium.Polygon(locations=poly.exterior.coords, color=colores[i], fill=True, fill_color=colores_claros[i], fill_opacity=0.4).add_to(m)
    
    for punto in grupo:
        folium.Marker(
            location=punto["coordenadas"],
            popup=tooltip_style.format(
                titulo=punto["titulo"],
                direccion=punto["direccion"],
                circuitos=''.join([f"<li>{circuito}</li>" for circuito in punto["circuitos"]])
            ),
            icon=folium.Icon(color=colores[i])
        ).add_to(m)
    
    # Calcular el centro del grupo y añadir el marcador del encargado
    centro = calcular_centro(grupo)
    total_circuitos = suma_circuitos(grupo)
    folium.Marker(
        location=centro,
        popup=f"<div style='font-family: Arial; font-size: 12pt;'>Encargado Cte. Mayor Juan Dos Santos<br>Total Policías = {total_circuitos}</div>",
        icon=folium.CustomIcon(icon_image='C:/PROYECTOS/mapaEleccionesJunio/movil.jpeg', icon_size=(30, 30))
    ).add_to(m)

# Añadir los puntos individuales
for punto in puntos_individuales:
    folium.Marker(
        location=punto["coordenadas"],
        popup=tooltip_style.format(
            titulo=punto["titulo"],
            direccion=punto["direccion"],
            circuitos=''.join([f"<li>{circuito}</li>" for circuito in punto["circuitos"]])
        ),
        icon=folium.Icon(color='gray')
    ).add_to(m)

# Guardar el mapa en un archivo HTML
map_path = "mapa_circuitos_grupos.html"
m.save(map_path)

print(f"Mapa guardado en {map_path}")