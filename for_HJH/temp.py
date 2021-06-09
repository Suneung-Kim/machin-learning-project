from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def get_point(target_size, font_size):

    return (target_size[1] // 2 - font_size[0] // 2, \
            target_size[0] // 2 - font_size[1] // 2) # W, H

class Text_to_Image:
    def __init__(self, cap, n_list=0):
        self.target_shape = (cap[0] // 10, cap[1], 3) # (H, W)
        self.target_shape = tuple(map(int, self.target_shape))
        self.font = ImageFont.truetype('/home/suneung/gulim.ttf', self.target_shape[0] // 1)
        self.font2 = ImageFont.truetype('/home/suneung/gulim.ttf', self.target_shape[0] // 1)
        self._generate_board()
        self._color = (0, 0, 0, 0)
        self.counter = 0
        self.n_list = n_list
        # self.point = (self.target_shape[1] // 2 - self.font.getsize(txt)[0] // 2, \
        #         self.target_shape[0] // 2 - self.font.getsize(txt)[1] // 2) # W, H
    
    def draw_mode(self, frame, name, id, txt = '시험 중, ', color=(0, 0, 0)):
        
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)

        if id != name.split('_')[0]:
            color = (0, 0, 255)
            txt += '본인 인증 실패, 현재 {} 감지됨'.format(name.split('_')[0])
        else:
            txt += '본인 인증 성공'
         
        draw.text(self.point, txt, font=self.font, fill=color)
        
        board = np.array(frame_pil)

        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

        return frame

    def easy_draw(self, frame, txt='', color=(0, 0, 0)): # frame, board -> frame + board (with text)

        self.point = get_point(self.target_shape, self.font.getsize(txt))

        board_pil = Image.fromarray(self.board)
        draw = ImageDraw.Draw(board_pil)
        draw.text(self.point, txt, font=self.font, fill=color)
        
        frame_pil = Image.fromarray(frame)
        
        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

        self._reset_board()

        return frame 


    def draw(self, frame, remain, attack=0.0, name='', id=''): # frame, board -> frame + board (with text)

        txt1 = "{}초 동안 인증이 진행됩니다.".format(remain)
        self.point = get_point(self.target_shape, self.font.getsize(txt1))


        board_pil = Image.fromarray(self.board)
        draw = ImageDraw.Draw(board_pil)
        draw.text(self.point, txt1, font=self.font, fill=(0, 0, 0, 0))
        
        frame_pil = Image.fromarray(frame)

        color = None
        if attack >= 0.6:
            txt2 = "Antispoofing Detected / "
            color = (0, 0, 255)
        else:
            txt2 = "Antispoofing is not Detected / "
            color = (0, 255, 0)

        if id != name.split('_')[0]:
            txt2 = "{} 인식 실패, 현재 {} 인식됨".format(self.n_list[id], self.n_list[name.split('_')[0]])
            color = (0, 0, 255)
            self.counter -= 1
        else:
            txt2 += "{} 인식 성공".format(self.n_list[id])
            if color is None:
                color = (0, 255, 0) 
            self.counter += 1


        # txt = "Antispoofing Detected" if attack >= 0.6 else "Antispoofing is not Detected, 현재 : {}".format(name)
        draw = ImageDraw.Draw(frame_pil)
        
        # if id == name.split('_')[0]:
        draw.text(point, txt2, font=self.font, fill=color)
        # else:
        #     draw.text(point, txt2, font=self.font, fill=color)
        frame = np.array(frame_pil)
        
        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

        self._reset_board()

        return frame, self.counter
    
    def await_for_time(self, thresh, msg='{}초 동안 대기합니다.', color=(0, 0, 0), out=None):
        cap = cv2.VideoCapture(-1)
        now = time.time()
        while time.time() - now < thresh:
            ok, frame = cap.read()
            self.point = get_point(self.target_shape, self.font.getsize(msg))

            broad_pil = Image.fromarray(self.board)
            draw = ImageDraw.Draw(broad_pil)
            draw.text(self.point, msg.format(int(thresh - (time.time() - now))), font=self.font, fill=color)

            # frame_pil = Image.fromarray(frame)
            # point = (self.target_shape[1] - self.font.getsize(msg.format(thresh - (time.time() - now)))[0], self.target_shape[0])
            # draw = ImageDraw.Draw(frame_pil)
            # draw.text(point, msg.format(thresh - (time.time() - now)), font=self.font, fill=(0, 255, 0, 0))
            # frame = np.array(frame_pil)

            frame[:self.target_shape[0], :self.target_shape[1]] = np.array(broad_pil)
            print(out)
            if out is not None:
                out.write(frame)
            cv2.imshow('Monitoring', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            self._reset_board()
        if out is not None:
            return out

        cap.release()

    def _generate_board(self,):
        self.board = np.zeros(self.target_shape, dtype=np.uint8)
        self.board[:] = (255, 255, 255)
    
    def _reset_board(self, ):
        self.board[:] = (255, 255, 255)