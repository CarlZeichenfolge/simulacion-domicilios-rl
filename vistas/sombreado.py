"""Shaders GLSL de la vista 3D.

Por qué propios y no los de Ursina: `lit_with_shadows_shader` ignora la luz
ambiente (de noche todo queda negro o, si se sube el ambiente, lavado) y no
permite cambiar el color de la luz con la hora. Aquí:

- iluminado: sol/luna con sombras suaves (PCF 3x3) + luz ambiente de "hemisferio"
  (lo que mira al cielo recibe el color del cielo; lo que mira al suelo, el del
  suelo) + niebla del color del horizonte, para que la maqueta se funda con el cielo.
- emisivo: sin iluminación (ventanas encendidas, faroles, lluvia, letreros), con
  textura opcional y la misma niebla.
- cielo: degradado cenit-horizonte, disco del sol, luna y estrellas, en una esfera
  que sigue a la cámara.

Los valores que cambian con la hora (colores, niebla, posición de la cámara) se
ponen UNA vez por cuadro en el nodo raíz de la escena y todos los heredan.
"""
from __future__ import annotations

from ursina.shader import Shader

_LUZ = """
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];
"""

_NIEBLA = """
uniform vec4 niebla_color;
uniform float niebla_inicio;
uniform float niebla_fin;
uniform vec3 camara_mundo;

vec3 con_niebla(vec3 color, vec3 pos_mundo) {
    float d = length(pos_mundo - camara_mundo);
    float t = clamp((d - niebla_inicio) / max(niebla_fin - niebla_inicio, 0.001), 0.0, 1.0);
    return mix(color, niebla_color.rgb, t * niebla_color.a);
}
"""

iluminado = Shader(language=Shader.GLSL, name="iluminado", vertex="""#version 150
""" + _LUZ + """
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 vertex;
in vec3 normal;
in vec4 p3d_Color;
out vec4 color_vertice;
out vec3 normal_vista;
out vec3 normal_mundo;
out vec3 pos_mundo;
out vec4 coord_sombra;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    vec4 pos_vista = p3d_ModelViewMatrix * vertex;
    normal_vista = normalize(p3d_NormalMatrix * normal);
    normal_mundo = normalize(mat3(p3d_ModelMatrix) * normal);
    pos_mundo = (p3d_ModelMatrix * vertex).xyz;
    coord_sombra = p3d_LightSource[0].shadowViewMatrix * pos_vista;
    color_vertice = p3d_Color;
}
""", fragment="""#version 150
""" + _LUZ + _NIEBLA + """
uniform vec4 p3d_ColorScale;
uniform vec3 luz_cielo;
uniform vec3 luz_suelo;
uniform float sombra_sesgo;
uniform float sombra_texel;
in vec4 color_vertice;
in vec3 normal_vista;
in vec3 normal_mundo;
in vec3 pos_mundo;
in vec4 coord_sombra;
out vec4 color_final;

void main() {
    vec4 base = color_vertice * p3d_ColorScale;
    vec3 n = normalize(normal_vista);
    vec3 hacia_luz = normalize(p3d_LightSource[0].position.xyz);
    float difusa = max(dot(n, hacia_luz), 0.0);

    float luz_directa = 0.0;
    if (difusa > 0.0) {
        vec4 c = coord_sombra;
        c.z -= sombra_sesgo * c.w;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec4 o = c;
                o.xy += vec2(float(x), float(y)) * sombra_texel * c.w;
                luz_directa += textureProj(p3d_LightSource[0].shadowMap, o);
            }
        }
        luz_directa /= 9.0;
    }
    float arriba = normalize(normal_mundo).y * 0.5 + 0.5;
    vec3 ambiente = mix(luz_suelo, luz_cielo, arriba);
    vec3 luz = ambiente + p3d_LightSource[0].color * difusa * luz_directa;
    color_final = vec4(con_niebla(base.rgb * luz, pos_mundo), base.a);
}
""")

emisivo = Shader(language=Shader.GLSL, name="emisivo", vertex="""#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 color_vertice;
out vec2 uv;
out vec3 pos_mundo;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    pos_mundo = (p3d_ModelMatrix * vertex).xyz;
    color_vertice = p3d_Color;
    uv = p3d_MultiTexCoord0;
}
""", fragment="""#version 150
""" + _NIEBLA + """
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
in vec4 color_vertice;
in vec2 uv;
in vec3 pos_mundo;
out vec4 color_final;

void main() {
    vec4 c = texture(p3d_Texture0, uv) * color_vertice * p3d_ColorScale;
    color_final = vec4(con_niebla(c.rgb, pos_mundo), c.a);
}
""")

cielo = Shader(language=Shader.GLSL, name="cielo", vertex="""#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 vertex;
out vec3 pos_mundo;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    pos_mundo = (p3d_ModelMatrix * vertex).xyz;
}
""", fragment="""#version 150
uniform vec3 camara_mundo;
uniform vec3 cielo_cenit;
uniform vec3 cielo_horizonte;
uniform vec3 cielo_bajo;
uniform vec3 sol_dir;
uniform vec3 sol_color;
uniform float sol_visible;
uniform vec3 luna_dir;
uniform float noche;
uniform float tiempo;
in vec3 pos_mundo;
out vec4 color_final;

float azar(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

void main() {
    vec3 d = normalize(pos_mundo - camara_mundo);
    float alto = clamp(d.y, 0.0, 1.0);
    vec3 color = mix(cielo_horizonte, cielo_cenit, pow(alto, 0.55));
    color = mix(color, cielo_bajo, clamp(-d.y * 3.0, 0.0, 1.0));

    // Sol: disco + halo grande que tiñe el horizonte de su lado
    float s = max(dot(d, normalize(sol_dir)), 0.0);
    color += sol_color * (pow(s, 900.0) * 3.0 + pow(s, 12.0) * 0.35) * sol_visible;

    // Luna y estrellas, solo de noche y solo arriba del horizonte
    float sobre = smoothstep(-0.02, 0.08, d.y);
    float l = max(dot(d, normalize(luna_dir)), 0.0);
    color += vec3(0.85, 0.9, 1.0) * (smoothstep(0.9993, 0.9996, l) * 0.9 + pow(l, 60.0) * 0.12) * noche * sobre;
    vec3 celda = floor(d * 180.0);
    float estrella = step(0.9975, azar(celda));
    float titilar = 0.6 + 0.4 * sin(tiempo * (1.5 + 3.0 * azar(celda + 1.0)) + 6.28 * azar(celda + 2.0));
    color += vec3(estrella * titilar) * noche * sobre * smoothstep(0.05, 0.35, d.y);

    color_final = vec4(color, 1.0);
}
""")
