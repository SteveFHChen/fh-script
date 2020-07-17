import wx
 

class App(wx.App):
 
    def OnInit(self): 
        frame = wx.Frame(parent=None, title='This is Frame Title')
 
        frame.Show()
 
        return True
 

app = App() 
app.MainLoop()

