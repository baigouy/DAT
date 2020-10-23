from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, QRectF
# logging
from dendritic_arborization_tracer.logger import TA_logger
logger = TA_logger()

class VectorialDrawPane:

    def __init__(self, active=False, demo=False, scale=1.0, drawing_mode=False):
        self.shapes = []
        self.currently_drawn_shape = None
        self.shape_to_draw = None
        self.selected_shape = []
        self.active = active
        self.scale = scale
        self.drawing_mode = drawing_mode


    def paintEvent(self, *args):
        painter = args[0]
        visibleRect = None
        if len(args) >= 2:
              visibleRect = args[1]

        painter.save()
        if self.scale != 1.0:
            painter.scale(self.scale, self.scale)

        for shape in self.shapes:
            # only draw shapes if they are visible --> requires a visiblerect to be passed
            if visibleRect is not None:
                # only draws if in visible rect
                if shape.boundingRect().intersects(QRectF(visibleRect)):
                    shape.draw(painter)
            else:
                shape.draw(painter)

        if self.currently_drawn_shape is not None:
            if self.currently_drawn_shape.isSet:
                self.currently_drawn_shape.draw(painter)

        sel = self.create_master_rect()
        if sel is not None:
            painter.drawRect(sel)
        painter.restore()

    def group_contains(self, x, y):
        # checks if master rect for group contains click
        # get bounds and create union and compare
        master_rect = self.create_master_rect()
        if master_rect is None:
            return False
        return master_rect.contains(QPointF(x, y))

    def create_master_rect(self):
        master_rect = None
        if self.selected_shape:
            for shape in self.selected_shape:
                if master_rect is None:
                    master_rect = shape.boundingRect()
                else:
                    master_rect = master_rect.united(shape.boundingRect())
        return master_rect

    def removeCurShape(self):
        if self.selected_shape:
            self.shapes = [e for e in self.shapes if e not in self.selected_shape]
            self.selected_shape = []

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos() / self.scale
            self.firstPoint = event.pos() / self.scale

            shapeFound = False
            if self.currently_drawn_shape is None:
                for shape in reversed(self.shapes):
                    if shape.contains(self.lastPoint) and not shape in self.selected_shape:
                        logger.debug('you clicked shape:' + str(shape))
                        if event.modifiers() == QtCore.Qt.ControlModifier:
                            if shape not in self.selected_shape:  # avoid doublons
                                self.selected_shape.append(shape)  # add shape to group
                                logger.debug('adding shape to group')
                                shapeFound = True
                        else:
                            if not self.group_contains(self.lastPoint.x(), self.lastPoint.y()):
                                self.selected_shape = [shape]
                                logger.debug('only one element is selected')
                                shapeFound = True
                        return

                if not shapeFound and event.modifiers() == QtCore.Qt.ControlModifier:
                    for shape in reversed(self.shapes):
                        if shape.contains(self.lastPoint):
                            if shape in self.selected_shape:  # avoid doublons
                                logger.debug('you clicked again shape:' + str(shape))
                                self.selected_shape.remove(shape)  # add shape to group
                                logger.debug('removing a shape from group')
                                shapeFound = True
                # no shape found --> reset sel
                if not shapeFound and not self.group_contains(self.lastPoint.x(), self.lastPoint.y()):
                    logger.debug('resetting sel')
                    self.selected_shape = []

            # check if a shape is selected and only move that
            if self.drawing_mode and not self.selected_shape and self.currently_drawn_shape is None:
                # do not reset shape if not done drawing...
                if self.shape_to_draw is not None:
                    self.currently_drawn_shape = self.shape_to_draw()
                else:
                    self.currently_drawn_shape = None
            if self.drawing_mode and not self.selected_shape:
                if self.currently_drawn_shape is not None:
                    self.currently_drawn_shape.set_P1(QPointF(self.lastPoint.x(), self.lastPoint.y()))

    def mouseMoveEvent(self, event):
        if event.buttons() and QtCore.Qt.LeftButton:
            if self.selected_shape and self.currently_drawn_shape is None:
                logger.debug('moving' + str(self.selected_shape))
                for shape in self.selected_shape:
                    shape.translate(event.pos() / self.scale - self.lastPoint)

        if self.currently_drawn_shape is not None:
            self.currently_drawn_shape.add(self.firstPoint, self.lastPoint)

        self.lastPoint = event.pos() / self.scale

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            if self.drawing_mode and self.currently_drawn_shape is not None:
                self.currently_drawn_shape.add(self.firstPoint, self.lastPoint)
                # if isinstance(self.currently_drawn_shape, Freehand2D):
                #     # this closes the freehand shape
                #     self.currently_drawn_shape.add(self.lastPoint, self.firstPoint)
                # # should not erase the shape if it's a polyline or a polygon by the way
                # if not isinstance(self.currently_drawn_shape, PolyLine2D) and not isinstance(self.currently_drawn_shape, Polygon2D):
                #     self.shapes.append(self.currently_drawn_shape)
                #     self.currently_drawn_shape = None

    def mouseDoubleClickEvent(self, event):
        # if isinstance(self.currently_drawn_shape, PolyLine2D) or isinstance(self.currently_drawn_shape, Polygon2D):
        #     self.shapes.append(self.currently_drawn_shape)
        #     self.currently_drawn_shape = None
        pass

if __name__ == '__main__':
    VectorialDrawPane()
