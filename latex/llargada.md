Extensió: l’extensió del TFG ha de ser entre 42.000 i 63.000 caràcters amb espais aproximadament, equivalent a 20–30 pàgines de dimensions convencionals, sense comptar la bibliografia ni els annexos.

Procediment de recompte:

- Compilo una còpia temporal del document amb `\tablesfalse` per excloure el text de les taules.
- Activo temporalment `\pagestyle{empty}` per evitar que les capçaleres i la paginació alterin el recompte.
- Converteixo el PDF resultant a text amb `pdftotext -enc UTF-8 main.pdf main.txt`.
- Selecciono el text des de `1. Introducció` fins al final de `Conclusions`, sense incloure les referències.
- Compto els caràcters amb `wc -m`.

Resultats:

- **62.014 caràcters** amb espais i salts de línia, tal com els compta `wc -m`.
- **61.716 caràcters** amb els espais normalitzats i sense comptar els salts de línia.
- **52.333 caràcters** sense espais ni salts de línia.
- **9.475 paraules**.

Per tant, el text principal queda dins del límit orientatiu de 42.000–63.000 caràcters amb espais.
