import pandas
import coremltools
from sklearn.linear_model import LinearRegression

rawdata = pandas.read_csv("cars.csv")

model = LinearRegression()
model.fit(rawdata[["modelo", "extras", "kilometraje", "estado"]], rawdata["precio"]) #El valor que se ajustara data["price"], corchetes dobles porque pandas sobreescribe uno de los dos corchetes, el mas interno significa un array, los externos los uso panda como operadores de indexacion que es la forma en la que lee los datos.

coreml_model = coremltools.converters.sklearn.convert(model, ["modelo", "extras", "kilometraje", "estado"], "precio")

coreml_model.author = "David Garcia"
coreml_model.license = "CCO"
coreml_model.short_description = "Este modelo predice el valor de venta de un coche segun diversos parametros"

coreml_model.save("Cars.mlmodel") #El nombre del save() sera el nombre de la clase de swift por eso empieza con mayusculas