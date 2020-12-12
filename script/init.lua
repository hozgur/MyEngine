
print ("MyEngine " .. engine.getversion())
print (_VERSION)

--luaNumpyIncludePath = "D:\DevTools\Anaconda\lib\site-packages\numpy\core\include"

function int(number)
	return math.floor(number)
end

color = {}

color.rgb = function(r,g,b)
	return (255 << 24) + (int(r) << 16 ) + (int(g) << 8 ) + int(b)
end

color.random = function( )
	return color.rgb(255 * math.random(),255 * math.random(), 255 * math.random())
end

color.red	= 0xFFFF0000
color.green = 0xFF00FF00
color.blue	= 0xFF0000FF
color.black = 0xFF000000
color.white = 0xFFFFFFFF


Key ={  NONE = 0,
        A = 1, B = 2, C = 3, D = 4, E = 5, F = 6, G = 7, H = 8, I = 9, J = 10, K = 11, L = 12, M = 13, N = 14, O = 15, P = 16, Q = 17, R = 18, S = 19, T = 20, U = 21, V = 22, W = 23, X = 24, Y = 25, Z = 26,
        K0 = 27, K1 = 28, K2 = 29, K3 = 30, K4 = 31, K5 = 32, K6 = 33, K7 = 34, K8 = 35, K9 = 36,
        F1 = 37, F2 = 38, F3 = 39, F4 = 40, F5 = 41, F6 = 42, F7 = 43, F8 = 44, F9 = 45, F10 = 46, F11 = 47, F12 = 48,
        UP = 49, DOWN = 50, LEFT = 51, RIGHT = 52,
        SPACE = 53, TAB = 54, SHIFT = 55, CTRL = 56, INS = 57, DEL = 58, HOME = 59, END = 60, PGUP = 61, PGDN = 62,
        BACK = 63, ESCAPE = 64, RETURN = 65, ENTER = 66, PAUSE = 67, SCROLL = 68,
        NP0 = 69, NP1 = 70, NP2 = 71, NP3 = 72, NP4 = 73, NP5 = 74, NP6 = 75, NP7 = 76, NP8 = 77, NP9 = 78,
        NP_MUL = 79, NP_DIV = 80, NP_ADD = 81, NP_SUB = 82, NP_DECIMAL = 83, PERIOD = 84
        }