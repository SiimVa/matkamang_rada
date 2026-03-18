# matkamang_rada

Streamliti rakendus Maa- ja Ruumiameti WFS andmetega rajakoridori ja katastri analüüsiks.

## Käivitamine

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Mida rakendus teeb

- teisendab sisestatud MGRS punktid L-EST97 koordinaatideks
- moodustab punktidest analüüsiala BBOX-i
- laeb ETAK ja katastri WFS kihid
- arvutab maastiku kokkuvõtlikud näitajad
- kuvab omandivormi jaotused
- joonistab kolm kaarti: analüüsiala, eraomand ja koondkaart

## Sisendid Streamlitis

- MGRS punktid tekstiväljas, üks punkt reale
- kaardikihtide sisse-välja lülitamine külgribal
- analüüsi käivitamine nupust
