# MLA 2025
The start of the greatest thesis of all time.



# Ideer til speciale:

## 
<details>
    <summary>Loss landscape analysis</summary>
Findes der en metode til at finde ud af hvor "rugged" ens landskab er. Kan man få en værdi på det? så man ved om ens model har svært ved at finde det globale minima
ChatGPT sagde:

Virkelig godt og vigtigt spørgsmål 🙌
Det du spørger ind til er faktisk et forskningsfelt i sig selv: “loss landscape analysis”. Hvordan måler vi, hvor “rugged” (ujævnt, med mange dale og toppe) landskabet er? Der findes ikke én perfekt værdi, men der er flere metoder og mål, folk bruger:

1. Hessian-analyse (krumning)

Hessianen er matrixen af alle andenordens-derivater af loss’en mht. parametrene.

Hvis mange egenværdier af Hessianen er negative og store i størrelse, har vi et rugged landskab (mange retninger med op- og nedbakker = sadler).

Hvis egenværdierne er små eller positive, er landskabet mere “glat”.
👉 Man kan måle spredningen af Hessianens egenværdier som en slags “ruggedness score”.

2. Sharpness vs. flatness af minima

“Skal jeg stå i en smal dal (sharp minimum) eller en bred dal (flat minimum)?”

Flat minima er mere stabile og generaliserer bedre.

Man kan måle dette ved at se på, hvor hurtigt loss’en vokser, når man bevæger sig lidt væk fra minimum i tilfældige retninger.
👉 Sharpness = “ruggedness” lokalt omkring minimum.

3. Random directions probing

Vælg en tilfældig retning i parameterrummet.

Bevæg dig langs denne retning og plott loss’en.

Rugged landskab → mange små toppe og dale i loss-kurven.

Smooth landskab → kurven er glat og monoton.
👉 Man kan kvantificere “ruggedness” ved fx variansen af gradienter langs tilfældige retninger.

4. Gradient noise scale

Hvis gradienter varierer meget (støjende), er landskabet rugged.

Hvis gradienterne peger nogenlunde i samme retning, er landskabet glat.
👉 Variansen i gradienter over batches kan bruges som mål.

5. Empiriske heuristikker

Learning rate finder: Hvis du kan bruge en høj læringsrate uden at divergere, er landskabet sandsynligvis ret fladt.

Loss surface visualization: Man kan lave 2D/3D plots ved at vælge to retninger i parameterrummet og “slice” landskabet. Det bruges tit i research.

Eksempel på praktiske “ruggedness-scorer”

Condition number af Hessianen (forhold mellem største og mindste positive egenværdi). Stor værdi → rugged.

Trace of Hessian (sum af egenværdier). Højt → rugged.

Variance of gradient norms mellem batches. Høj → rugged.

🔑 Opsummering:
Der findes ikke ét “tal” der fortæller, hvor rugged et landskab er. Men man kan:

Analysere Hessianen (krumning).

Måle sharpness/flatness af minima.

Kigge på gradientstøj og variation.

Bruge små probes i tilfældige retninger.

I praksis er det computermæssigt dyrt at beregne præcist i store netværk (fordi Hessianen er gigantisk), men man kan lave approksimationer og få en fornemmelse.
</details>