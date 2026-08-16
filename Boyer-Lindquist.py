import math
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from colour import Flux_funcr, bb_to_rgb


vertex_shader = """
#version 330 core

void main() {

    // triangle clip space for full screen
    vec2 pos;

    if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
    if (gl_VertexID == 1) pos = vec2( 3.0, -1.0);
    if (gl_VertexID == 2) pos = vec2(-1.0,  3.0);

    gl_Position = vec4(pos, 0.0, 1.0);
}
"""
fragment_shader = """
#version 330 core

uniform float M;
uniform float a;
uniform float tanHalfFov;   //for camera resolution
uniform vec2 resolution;
uniform float r_cam, theta_cam, phi_cam;   // camera event
uniform float DISC_IN;
uniform float inner_photon_orbit;
uniform float outer_photon_orbit;
out vec4 fragColor;

const int   MAX_STEPS = 3000; 
const float H_STEP    = 0.1;
const float R_ESCAPE  = 120.0;
const float DISC_OUT  = 20.0;
const int loiter_max = 50; 
const float radial_drift = 0.07;
uniform sampler1D discTemp;
uniform sampler1D bbColor;
uniform float T_LUT_MIN;
uniform float T_LUT_MAX;
uniform float exposure;
uniform float T_peak;
uniform sampler2D starfield;



struct State { 
    vec4 x;
    vec4 p;
};

float Sigma(float r, float a, float theta) {
    return r*r + a*a*cos(theta)*cos(theta);
}

float Delta(float r, float M, float a) {
    return r*r - 2.0*M*r + a*a;
}

// formula for xdot ^mu  = g^munu p_nu 

float gtt(float r, float theta) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    return -((r*r+a*a)*(r*r+a*a) - a*a*d*sin(theta)*sin(theta)) / (s*d);
}

float gtphi(float r, float theta) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    return - 2.0*M*a*r / (s*d);
}

float grr(float r, float theta) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    return d/s;
}

float gthetatheta(float r, float theta) {
    float s = Sigma(r, a, theta);
    return 1.0 / s;
}

float gphiphi(float r, float theta) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float st = sin(theta);
    float st2 = max(st*st, 1e-4);
    return (d - a*a*st2) / (s*d*st2);
}

vec4 xdot(State s) {
    float t = s.x.x;
    float r = s.x.y;
    float theta = s.x.z;
    float phi = s.x.w;

    float p_t = s.p.x;  //Tetrad intialization feeds these ab - initio
    float p_r = s.p.y;
    float p_theta = s.p.z;
    float p_phi = s.p.w;

    return vec4(
        gtt(r, theta) * p_t + gtphi(r, theta) * p_phi,
        grr(r, theta) * p_r,
        gthetatheta(r, theta) * p_theta,
        gtphi(r, theta) * p_t + gphiphi(r, theta) * p_phi
    );
}

//pdot has a lot of helper functions that build it, due to the partial derivitves of metric componenets.
// remmeber that pdot_t and pdot_phi are zero due to killing field symmetries.

float Sigma_r(float r) {
    return 2.0*r;
}

float Sigma_theta(float r, float a, float theta) {
    return -2.0*a*a*cos(theta)*sin(theta);
}

float Delta_r(float r, float M, float a) {
    return 2.0*r - 2.0*M;
}
// Delta_theta is zero

float B(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    return s*d;
}

float B_r(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float s_r = Sigma_r(r);
    float d_r = Delta_r(r, M, a);
    return s_r*d + s*d_r;
}

float B_theta(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float s_theta = Sigma_theta(r, a, theta);
    // Delta_theta is zero
    return s_theta*d;
}

float delr_gtt(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float s_r = Sigma_r(r);
    float d_r = Delta_r(r, M, a);
    float b = B(r, a, theta, M);
    float b_r = B_r(r, a, theta, M);
    float N   = (r*r+a*a)*(r*r+a*a) - a*a*d*sin(theta)*sin(theta);
    float N_r = 4.0*r*(r*r+a*a) - a*a*d_r*sin(theta)*sin(theta);
    return (-N_r*b + N*b_r) / (b*b);      // this one is hideous, hence compaction.
}

float delr_gtphi(float M, float r, float a, float theta) {
    float b = B(r, a, theta, M);
    float b_r = B_r(r, a, theta, M);
    return -2.0*M*a/b + 2.0*M*a*r*b_r/(b*b);
}

float delr_grr(float r, float a, float theta, float M) {
    float s   = Sigma(r, a, theta);
    float d   = Delta(r, M, a);
    float s_r = Sigma_r(r);
    float d_r = Delta_r(r, M, a);
    return (d_r*s - d*s_r) / (s*s);
}

float delr_gthetatheta(float r, float a, float theta, float M) {
    float s   = Sigma(r, a, theta);
    float s_r = Sigma_r(r);
    return -s_r / (s*s);
}

float delr_gphiphi(float r, float a, float theta, float M) {
    float s   = Sigma(r, a, theta);
    float d   = Delta(r, M, a);
    float s_r = Sigma_r(r);
    float d_r = Delta_r(r, M, a);
    float st2 = sin(theta)*sin(theta); //one of the hardest derivitives, just alot of compaction.
    float t1 = -s_r / (s*s*st2);                
    float t2 =  a*a*(s_r*d + s*d_r) / (s*s*d*d); 
    return t1 + t2;
}

// theta partials from here down

float deltheta_gtt(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float s_theta = Sigma_theta(r, a, theta);
    float b = B(r, a, theta, M);
    float b_theta = B_theta(r, a, theta, M);
    float N       = (r*r+a*a)*(r*r+a*a) - a*a*d*sin(theta)*sin(theta);
    float N_theta = -a*a*d*2.0*sin(theta)*cos(theta);   // ∂θ of -a²Δsin²θ
    return (-N_theta*b + N*b_theta) / (b*b);
}

float deltheta_gtphi(float r, float a, float theta, float M) {
    float b = B(r, a, theta, M);
    float b_theta = B_theta(r, a, theta, M);
    return b_theta * 2.0*M*a*r / (b*b);
}

float deltheta_grr(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float s_theta = Sigma_theta(r, a, theta);
    float d = Delta(r, M, a);
    return -s_theta*d / (s*s);
}

float deltheta_gthetatheta(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float s_theta = Sigma_theta(r, a, theta);
    return -s_theta / (s*s);   
}

float deltheta_gphiphi(float r, float a, float theta, float M) {
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float st = sin(theta);
    float st2 = max(st*st, 1e-4);
    float ct = cos(theta);
    float s_theta = Sigma_theta(r, a, theta);   // -2 a^2 sinθ cosθ

    float t1 = -(s_theta*st2 + 2.0*s*st*ct) / (s*s*st2*st2);
    float t2 =  a*a*s_theta / (s*s*d);

    return t1 + t2;
}

vec4 pdot(State s) {
    float p_t = s.p.x;
    float p_r = s.p.y;
    float p_theta = s.p.z;
    float p_phi = s.p.w;

    float r = s.x.y;
    float theta = s.x.z; 

    return vec4(
        0.0, // killing field symmetries
        -0.5 * (delr_gtt(r, a, theta, M) * p_t*p_t + 2.0*delr_gtphi(M,r, a, theta) * p_t*p_phi + delr_grr(r, a, theta, M) * p_r*p_r + delr_gthetatheta(r, a, theta, M) * p_theta*p_theta + delr_gphiphi(r, a, theta, M) * p_phi*p_phi),
        -0.5 * (deltheta_gtt(r, a, theta, M) * p_t*p_t + 2.0*deltheta_gtphi(r, a, theta, M) * p_t*p_phi + deltheta_grr(r, a, theta, M) * p_r*p_r + deltheta_gthetatheta(r, a, theta, M) * p_theta*p_theta + deltheta_gphiphi(r, a, theta, M) * p_phi*p_phi),
        0.0 // killing field symmetries
    );
}

struct dots {
    vec4 dx;
    vec4 dp;
};

dots dot_functions(State s) {
    dots d;
    d.dx = xdot(s);
    d.dp = pdot(s);
    return d;
}

State rk4step(State s, float h) {
    dots k1 = dot_functions(s);

    State s2;
    s2.x = s.x + 0.5*h*k1.dx;
    s2.p = s.p + 0.5*h*k1.dp;
    dots k2 = dot_functions(s2);

    State s3;
    s3.x = s.x + 0.5*h*k2.dx;
    s3.p = s.p + 0.5*h*k2.dp;
    dots k3 = dot_functions(s3);

    State s4;
    s4.x = s.x + h*k3.dx;
    s4.p = s.p + h*k3.dp;
    dots k4 = dot_functions(s4);

    s.x += (h/6.0)*(k1.dx + 2.0*k2.dx + 2.0*k3.dx + k4.dx);
    s.p += (h/6.0)*(k1.dp + 2.0*k2.dp + 2.0*k3.dp + k4.dp);

    return s;
}
// Redshift helper functions

vec3 sRGB_encode(vec3 u) {
    u = clamp(u, 0.0, 1.0);
    return mix(12.92*u, 1.055*pow(u, vec3(1.0/2.4)) - 0.055, step(0.0031308, u));
}

float redshift(float r, float E_now, float Lz, float M, float a) {               
    float rt = sqrt(r);
    float rM = sqrt(M);
    float Omeg_keplr = rM / (rt*r + a*rM);
    //using modified metric components because theta is in equatorial plane.
    float g_tt = -(1.0 - 2.0*M / r);
    float g_tphi = -2.0*M*a / r;
    float g_phiphi = r*r + a*a + 2.0*M *a*a/r;
    float denom = -(g_tt + 2.0*g_tphi*Omeg_keplr + g_phiphi*Omeg_keplr*Omeg_keplr);
    float ut_em = inversesqrt(max(denom, 1.0e-6)); //guard, shouldnt be an issue but it extends to NaN issues.
    float b = Lz / E_now;
    return 1.0 / (ut_em*(1.0 - Omeg_keplr*b));
}
//starfield helpers, normalized escape direction, etc

vec3 escapeDir(float r, float theta, float phi, float pr, float ptheta, float pphi) {
    float st  = max(sin(theta), 1e-4);
    float sph = sin(phi);
    float ct  = cos(theta);
    float cph = cos(phi);
    vec3 loc_esc = normalize(vec3(pr, ptheta / r, pphi / (r*st)));   // wrap in vec3()
    vec3 rhat = vec3(st*cph, st*sph,  ct);
    vec3 that = vec3(ct*cph, ct*sph, -st);
    vec3 phat = vec3(-sph,    cph,     0.0);
    return normalize(loc_esc.x*rhat + loc_esc.y*that + loc_esc.z*phat);
}

vec2 dirToUV(vec3 d) {
    float u = 0.5 + atan(d.y, d.x) / (2.0 * 3.141592654);
    float v = acos(clamp(d.z, -1.0, 1.0)) / 3.141592654;
    return vec2(u, v);
}

// next step here is to intialize the photons, which ive been putting off haha.

float A_func(float r, float a, float theta, float M) {
    float d = Delta(r, M, a);
    return (r*r+a*a)*(r*r+a*a) - a*a*d*sin(theta)*sin(theta);
}

vec4 init_momentum(vec4 x, vec2 uv) {
    float r = x.y;
    float theta = x.z;
    // remember that t and phi momenta are conserved.
    float s = Sigma(r, a, theta);
    float d = Delta(r, M, a);
    float A = A_func(r, a, theta, M);
    float Omega = 2.0*M*a*r / A;
    float alpha = sqrt((s*d) / A);
    
    vec3 direction = normalize(vec3(uv.x * tanHalfFov, uv.y * tanHalfFov, -1.0));
    float n_r = direction.z; // there is bug possibility here, n_r needs to remain <0
    float n_theta = -direction.y; 
    float n_phi = direction.x;

    //tetrad time!!!!!!!!!!!!!!!!!!!!!!!!!!!
    float E_loc = 1.0; //Not gonna change much.
    float p_t = -E_loc*(alpha + n_phi*Omega*sqrt(A/s)*sin(theta));
    float p_r =  E_loc*n_r*sqrt(s/d);
    float p_th =  E_loc*n_theta*sqrt(s);
    float p_ph =  E_loc*n_phi*sqrt(A/s)*sin(theta);

    return vec4(p_t, p_r, p_th, p_ph);
}

void main() {
    vec2 uv = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    uv.x *= resolution.x / resolution.y; 

    float r_plus = M + sqrt(M*M - a*a); // outer horizon

    State s;
    s.x = vec4(0.0, r_cam, theta_cam, phi_cam);
    s.p = init_momentum(s.x, uv);

    vec3 color = vec3(0.0);   //horizon/captured = black
    bool done = false;
    int loiter = 0;
    float r_loiter = s.x.y;

    for (int i = 0; i < MAX_STEPS && !done; i++) {
        float r_prev     = s.x.y;
        float theta_prev = s.x.z;

        s = rk4step(s, H_STEP);

        float r_now     = s.x.y;
        float theta_now = s.x.z;
        float Phi_now = s.x.w;
        bool in_shell = (r_now > inner_photon_orbit && r_now < outer_photon_orbit);
        float p_r_now = s.p.y;
        float p_theta_now = s.p.z;
        float E_now = -s.p.x;
        float Lz = s.p.w;

        // bugfixer for  the green lines along BL singularity and Photon sphere. Fated NAN's occur in rk4, just call em black
        if (isnan(r_now) || isinf(r_now) || isnan(s.x.z) || isnan(s.p.y)) {
        color = vec3(0.0); 
        done = true;
        }

        // 1) horizon capture
        if (r_now < r_plus + 0.01) {
            color = vec3(0.0);
            done = true;
        }
        // Disc model
        else if ((theta_prev - 1.5707963) * (theta_now - 1.5707963) < 0.0) {
            
            float f = (1.5707963 - theta_prev) / (theta_now - theta_prev);
            float r_cross = mix(r_prev, r_now, f); //Both theta and r are linear in h, so mix function can interpolate same affine parameter
            if (r_cross > DISC_IN && r_cross < DISC_OUT) {
                float u_r = (r_cross - DISC_IN) / (DISC_OUT - DISC_IN);
                float T   = texture(discTemp, u_r).r;
                
                //Redshift
                float g = redshift(r_cross, E_now, Lz, M, a); 
                float T_obs  = g * T;
                float u_T = clamp((T_obs - T_LUT_MIN) / (T_LUT_MAX - T_LUT_MIN), 0.0, 1.0);
                vec3  hue = texture(bbColor, u_T).rgb;
                float brt = pow(T_obs / T_peak, 4.0);          
                color = hue * brt * exposure;              
                done = true;
            }
        }
        // 3) escape to background
        else if (r_now > R_ESCAPE) {
            vec3 dir = escapeDir(r_now, theta_now, Phi_now, p_r_now, p_theta_now, Lz);  
            color    = texture(starfield, dirToUV(dir)).rgb * 250.0;
            done = true;
        }
        // 4) loiter checker, if photons are asymptoting around shadow, they incorporate.

        else if (in_shell) {
            loiter++;
            if (loiter >= loiter_max) {
                if (abs(r_now - r_loiter) < radial_drift) {
                    color = vec3(1.1, 0.0, 0.0);
                    done = true;
                }
                r_loiter = r_now;
                loiter = 0;
            } 
        }
        else {
            loiter = 0;
            r_loiter = r_now;
        }
    
    }

    // 4) ran out of steps without resolving — flag it
    if (!done) {
        color = vec3(0.0, 1.0, 0.0);   // bright green = "MAX_STEPS hit", debug only
    }
    // luminance Reinhard (preserves hue) -> sRGB
    float Lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (Lum > 0.0) color *= (Lum / (1.0 + Lum)) / Lum;
    color = sRGB_encode(color);
    fragColor = vec4(color, 1.0);
}
"""


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"{name} shader compile error:\n{error}")
    return shader


def create_program(vertex_source, fragment_source):
    vertex = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"shader link error:\n{error}")
    glDeleteShader(vertex)
    glDeleteShader(fragment)
    return program


def r_ISCO_prograde(M_val, a_val):
    Z1 = 1.0 + math.pow(1.0 - (a_val**2 / M_val**2), 1.0/3.0) * (
        math.pow(1.0 + (a_val / M_val), 1.0/3.0) + math.pow(1.0 - (a_val / M_val), 1.0/3.0))
    Z2 = math.pow((3.0 * a_val**2 / M_val**2) + Z1**2, 1.0/2.0)
    assert Z1 <= Z2, "Z2 >= Z1 for prograde orbits"
    return M_val * (3.0 + Z2 - math.pow((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2), 1.0/2.0))


def r_ph(M_val, a_val, prograde=True):
    s = -1.0 if prograde else 1.0
    return 2.0 * M_val * (1.0 + math.cos((2.0/3.0) * math.acos(s * a_val / M_val)))


def main():
    #physical parameters, update these and image & physics changes.
    M_val   = 1.0
    a_val   = 0.9
    fov_deg = 40.0
    r_camera     = 40.0
    theta_camera = math.radians(85.0)   # just above the equatorial plane
    phi_camera   = 0.0
    T_peak = 17000.0 
    WIDTH, HEIGHT = 800, 440

    if not glfw.init():
        raise RuntimeError("glfw init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE) 

    window = glfw.create_window(WIDTH, HEIGHT, "Kerr raytracer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1) 

    program = create_program(vertex_shader, fragment_shader)

    #attributeless, no VBO.
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glUseProgram(program)

    
    tan_half_fov = math.tan(0.5 * math.radians(fov_deg))
    disc_in   = r_ISCO_prograde(M_val, a_val)
    rph_inner = min(r_ph(M_val, a_val, True), r_ph(M_val, a_val, False))
    rph_outer = max(r_ph(M_val, a_val, True), r_ph(M_val, a_val, False))
    N_r = 1024
    r_grid = np.linspace(disc_in, 20.0, N_r) #20.0 is the oter disc edeg, currently its a cosnt float in the shader
    F_tilda = Flux_funcr(r_grid, M_val, a_val, 1.00)  
    T_scale = T_peak / (F_tilda.max()**(1/4)) 
    T_r = (F_tilda**(1/4) * T_scale).astype(np.float32) # Boltzmann

    T_LUT_MIN = 1000.0
    T_LUT_MAX = 20000.0
    T_LUT_N = 1024
    T_grid = np.linspace(T_LUT_MIN, T_LUT_MAX, T_LUT_N)
    blackbodtoRGB = np.ascontiguousarray([bb_to_rgb(T) for T in T_grid], dtype=np.float32) #This array type is a shader thing

    sky = np.load("starfield.npy")
    sky = np.ascontiguousarray(sky, np.float32)    # GL needs contiguous row-major bytes
    texH, texW = sky.shape[:2]

    starTex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, starTex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, texW, texH, 0, GL_RGB, GL_FLOAT, sky)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)         
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)  
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) # only doing this one once, no need for a function


    def make_1d_tex(data, internal_fmt, fmt):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, tex)
        glTexImage1D(GL_TEXTURE_1D, 0, internal_fmt, data.shape[0], 0, fmt, GL_FLOAT, data)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE) 
        return tex

    tex_disc = make_1d_tex(T_r, GL_R32F, GL_RED)
    tex_bb   = make_1d_tex(blackbodtoRGB, GL_RGB32F, GL_RGB)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_1D, tex_disc)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_1D, tex_bb)
    glActiveTexture(GL_TEXTURE2)   
    glBindTexture(GL_TEXTURE_2D, starTex)

    def loc(name):
        return glGetUniformLocation(program, name)

    glUniform1f(loc("M"), M_val)
    glUniform1f(loc("a"), a_val)
    glUniform1f(loc("tanHalfFov"), tan_half_fov)
    glUniform1f(loc("r_cam"), r_camera)
    glUniform1f(loc("theta_cam"), theta_camera)
    glUniform1f(loc("phi_cam"), phi_camera)
    glUniform1f(loc("DISC_IN"), disc_in)
    glUniform1f(loc("inner_photon_orbit"), rph_inner)
    glUniform1f(loc("outer_photon_orbit"), rph_outer)
    glUniform1i(loc("discTemp"), 0) 
    glUniform1i(loc("bbColor"),  1)  
    glUniform1f(loc("T_LUT_MIN"), T_LUT_MIN)
    glUniform1f(loc("T_LUT_MAX"), T_LUT_MAX)
    glUniform1f(loc("exposure"), 2.5)   #1<n<4
    glUniform1f(loc("T_peak"), T_peak)  
    glUniform1i(loc("starfield"), 2) 

    
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, fb_w, fb_h)
    glUniform2f(loc("resolution"), float(fb_w), float(fb_h))

    print("rendering frame, may take a sec")
    glClear(GL_COLOR_BUFFER_BIT)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glfw.swap_buffers(window)
    print("frame done")

    while not glfw.window_should_close(window):
        glfw.wait_events_timeout(0.1)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glDeleteVertexArrays(1, [vao])
    glDeleteProgram(program)
    glfw.terminate()
    print("window closed")


if __name__ == "__main__":
    main()