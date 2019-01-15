import math
######
class Colors():
    def __init__(self):
        self.colors=[]
        self.color_table={}
        self.color_table_key = []
        self.color_table_value = []

    def hsv2rgb(self,color_hsv):
        h = float(color_hsv[0])
        s = float(color_hsv[1])
        v = float(color_hsv[2])
        h60 = h / 60.0
        h60f = math.floor(h60)
        hi = int(h60f) % 6
        f = h60 - h60f
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = 0, 0, 0
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        elif hi == 5: r, g, b = v, p, q
        #r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return (b, g, r)

    def rgb2hsv(self,color_bgr):
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = df/mx
        v = mx
        return (h, s, v)

    def HSVDistance(self,hsv_1,hsv_2):
        H_1,S_1,V_1 = hsv_1
        H_2,S_2,V_2 = hsv_2
        # R=100
        # angle=30
        # h = R * math.cos(angle / 180 * math.pi)
        # r = R * math.sin(angle / 180 * math.pi)
        x1 =  V_1 * S_1 * math.cos(H_1 / 180 * math.pi)
        y1 =  V_1 * S_1 * math.sin(H_1 / 180 * math.pi)
        # z1 = h * (1 - V_1)
        x2 =  V_2 * S_2 * math.cos(H_2 / 180 * math.pi)
        y2 =  V_2 * S_2 * math.sin(H_2 / 180 * math.pi)
        # z2 = h * (1 - V_2)
        dx = x1 - x2
        dy = y1 - y2
        # dz = z1 - z2
        return math.sqrt(dx * dx + dy * dy )


    def ColourDistance(self,rgb_1, rgb_2):
        B_1, G_1, R_1 = rgb_1
        B_2, G_2, R_2 = rgb_2
        rmean = (R_1 + R_2) / 2
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2
        return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

    def color_140(self):
        self.color_table = {
        'aliceblue':            '#F0F8FF',
        'antiquewhite':         '#FAEBD7',
        'aqua':                 '#00FFFF',
        'aquamarine':           '#7FFFD4',
        'azure':                '#F0FFFF',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'black':                '#000000',
        'blanchedalmond':       '#FFEBCD',
        'blue':                 '#0000FF',
        'blueviolet':           '#8A2BE2',
        'brown':                '#A52A2A',
        'burlywood':            '#DEB887',
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',
        'coral':                '#FF7F50',
        'cornflowerblue':       '#6495ED',
        'cornsilk':             '#FFF8DC',
        'crimson':              '#DC143C',
        'cyan':                 '#00FFFF',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'floralwhite':          '#FFFAF0',
        'forestgreen':          '#228B22',
        'fuchsia':              '#FF00FF',
        'gainsboro':            '#DCDCDC',
        'ghostwhite':           '#F8F8FF',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'indigo':               '#4B0082',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'lime':                 '#00FF00',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navajowhite':          '#FFDEAD',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olive':                '#808000',
        'olivedrab':            '#6B8E23',
        'orange':               '#FFA500',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'silver':               '#C0C0C0',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'teal':                 '#008080',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'wheat':                '#F5DEB3',
        'white':                '#FFFFFF',
        'whitesmoke':           '#F5F5F5',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'}

        for key, value in self.color_table.items():
            digit = list(map(str, range(10)))+ list("ABCDEF")
            a1 = digit.index(value[1]) * 16 + digit.index(value[2])
            a2 = digit.index(value[3]) * 16 + digit.index(value[4])
            a3 = digit.index(value[5]) * 16 + digit.index(value[6])
            value = self.rgb2hsv((a3/255.,a2/255.,a1/255.))
            self.color_table[key]=value
        ###change dict to list
        self.color_table_key = list(self.color_table.keys())
        self.color_table_value = list(self.color_table.values())
    
    def color_10(self):
        color_list=[(100,75,68),(185,180,184),(152,139,123),(219,219,182),(109,107,110),
            (217,181,183),(182,155,146),(219,219,221),(110,109,73),(148,110,109),(0,0,255)]
        for color in color_list:
            a1=color[0]/255.
            a2=color[1]/255.
            a3=color[2]/255.
            c=(a3,a2,a1)
            self.colors.append(self.rgb2hsv(c))

    def find_nearest_color(self,color_BGR):
        color_hsv=self.rgb2hsv(color_BGR)
        color_diff_list = []
        #print(self.color_table_value)
        for color_f in self.colors:
            #colors_hsv=rgb2hsv(color_f)
            color_diff = self.HSVDistance(color_f, color_hsv)
            color_diff_list.append(color_diff)
        color_index = color_diff_list.index(min(color_diff_list))
        nearest_color = self.hsv2rgb(self.colors[color_index])
        return nearest_color

