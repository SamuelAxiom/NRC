import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image
from math import exp

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	data['damagelevel']=data['damagelevel'].apply(lambda x:'No damage' if x=='0' else x)
	data['last_nfi']=data['last_nfi'].apply(lambda x:'Never received' if x=='0' else x)
	data['treatment']=data['treatment'].apply(lambda x:'No treatment' if x=='0' else x)
	for i in ['challenge_1','challenge_2','challenge_3']:
		data[i]=data[i].apply(lambda x: x if data[i].value_counts()[x]>24 else 'Other')
	correl=pd.read_csv('graphs.csv',sep='\t')
	questions=pd.read_csv('questions.csv',sep='\t')
	questions.drop([i for i in questions.columns if 'Unnamed' in i],axis=1,inplace=True)
	quest=questions.iloc[3].to_dict()
	codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	return data,correl,quest,codes

data,correl,questions,codes=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

@st.cache
def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=['#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1','#9e9ac8',\
    		'#bcbddc','#dadaeb','#636363','#969696','#bdbdbd','#d9d9d9','#3182bd','#6baed6','#9ecae1','#c6dbef','#e6550d']
    link_colors=['#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1','#9e9ac8',\
    		'#bcbddc','#dadaeb','#636363','#969696','#bdbdbd','#d9d9d9','#3182bd','#6baed6','#9ecae1','#c6dbef','#e6550d']
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    #st.write(labels)
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #st.write(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #st.write(L,len(L),iteration)
    #st.write(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig


def count2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T*100
    agg2=agg2.astype(int)
    
    if abscisse=='district':
    	agg=agg.reindex(dataf['district'].unique().tolist())
    	agg2=agg2.reindex(dataf['district'].unique().tolist())
    
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig

def pourcent2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    
    if abscisse=='district':
    	agg=agg.reindex(dataf['district'].unique().tolist())
    	agg2=agg2.reindex(dataf['district'].unique().tolist())
    	
    x=agg2.index
    
    
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
    else:
        #st.write(agg)
        #st.write(agg2)
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig


questions=pd.read_csv('questions.csv',sep='\t')
correlations=pd.read_csv('correlations.csv',sep='\t')
questions=questions[[i for i in questions.columns if 'Unnamed' not in i]]
codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
continues=pickle.load( open( "cont_feat.p", "rb" ) )
cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
dummy_cols=pickle.load( open( "dummy.p", "rb" ) )	
questions.set_index('Idquest',inplace=True)
correl=pd.read_csv('graphs.csv',sep='\t')


text=[i for i in questions.columns if questions[i]['Treatment']=='text']
text2=[questions[i]['question'] for i in text if 'recomm' not in i]+['Recommandation progamming','Recommandation activities'] 
#st.write(text)

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoNRC.png")

def main():	
	cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('What do you want to do ?',('Mapping apps','Display prices correlations','Display Districts correlations','Display Education correlations','Display CSI or FCS correlations','Display Security correlations','Display Other correlations','Display Sankey diagrams'))
	
	title1, title3 = st.columns([9,2])
	title3.image(img2)
	
	if topic in ['Display prices correlations','Display Districts correlations']:
	
		if topic == 'Display prices correlations':
			title1.title('Correlations related to cost of living per district')
			st.markdown("""---""")	
			st.subheader('Cost of living and incomes are very much different from one district to another as we can see below:')
			st.markdown("""---""")	
			quest=correl[correl['topic']=='price']
		else: 
			title1.title('Correlations related to Districts (Not counting prices and incomes)')
			st.markdown("""---""")	
			st.subheader('On some aspects some districts are very different from the others:')
			st.markdown("""---""")	
			quest=correl[correl['topic']=='dist']
					
				
		df=data[['district']+quest['variable_y'].unique().tolist()].copy()
		
		for i in range(len(quest['variable_y'].unique())):
			
			if quest.iloc[i]['graphtype']=='violin':
				districts = df['district'].unique().tolist()
				fig=go.Figure()
				for district in districts:
					
					fig.add_trace(go.Violin(x=df['district'][df['district'] == district],
	                            		y=df[quest.iloc[i]['variable_y']][df['district'] == district],
	                            		name=district,
	                            		box_visible=True,
                           			meanline_visible=True,points="all",))
				fig.update_layout(showlegend=False)
				fig.update_yaxes(range=[-0.1, quest.iloc[i]['max']],title=quest.iloc[i]['ytitle'])
				st.subheader(quest.iloc[i]['title'])
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])	
			elif quest.iloc[i]['graphtype']=='bar':
				st.subheader(quest.iloc[i]['title'])
				
				col1,col2=st.columns([1,1])

				fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
				fig1.update_layout(title_text=quest.iloc[i]['legendtitle'],font=dict(size=12),showlegend=False)	
				col1.plotly_chart(fig1,use_container_width=True)
						
				fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,legendtitle='',xaxis=quest.iloc[i]['xtitle'])
				fig1.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
				col2.plotly_chart(fig2,use_container_width=True)
				st.write(quest.iloc[i]['description'])
			
				
			st.markdown("""---""")	
	
	elif topic in ['Display Education correlations','Display CSI or FCS correlations','Display Security correlations','Display Other correlations']:
		#st.write(cat_cols)
		if topic == 'Display Education correlations':
			title1.title('Correlations related to education')
			st.markdown("""---""")	
			
			quest=correl[correl['topic']=='educ']
		elif topic=='Display CSI or FCS correlations': 
			title1.title('Correlations related to food security scores (FCS or CSI)')
			st.markdown("""---""")	
			
			quest=correl[correl['topic']=='CSI']
		elif topic=='Display Security correlations': 
			title1.title('Correlations related to security')
			st.markdown("""---""")	
				
			quest=correl[correl['topic']=='secu']
		else:
			title1.title('Other correlations')
			st.markdown("""---""")	
			
			quest=correl[correl['topic']=='other']
		
		for i in range(len(quest)):
			
			if quest['variable_x'].iloc[i] in cat_cols and quest['variable_y'].iloc[i] in cat_cols:
				q1=quest['variable_x'].iloc[i]
				q2=quest['variable_y'].iloc[i]
				df=pd.DataFrame(columns=[q1,q2])
				quests1=[feat for feat in data.columns if q1 in feat[:len(q1)]]
				catq1=[' '.join(feat.split(' ')[1:]) for feat in quests1]
		
				#st.write(quests1)
				quests2=[feat for feat in data.columns if q2 in feat[:len(q2)]]
				catq2=[' '.join(feat.split(' ')[1:]) for feat in quests2]
				#st.write(quests2)
				#st.write(dfm[quests1+quests2])
				for feat1 in range(len(quests1)):
					for feat2 in range(len(quests2)):       
						ds=data[[quests1[feat1],quests2[feat2]]].copy()
						ds=ds[ds[quests1[feat1]]==1]
						ds=ds[ds[quests2[feat2]]==1]
						#st.write(ds)      
						ds[q1]=ds[quests1[feat1]].apply(lambda x: catq1[feat1])
						ds[q2]=ds[quests2[feat2]].apply(lambda x: catq2[feat2])
						#st.write(ds)
						df=df.append(ds[[q1,q2]])
				#st.write(df)
			
			elif quest['variable_x'].iloc[i] in cat_cols or quest['variable_y'].iloc[i] in cat_cols:
				if quest['variable_x'].iloc[i] in cat_cols:
				#st.write(cat_cols)
					cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
				else:
					cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
				df=pd.DataFrame(columns=[cat,autre])
					
				
				catcols=[j for j in data.columns if cat in j[:len(cat)]]
				cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
				#st.write(cats)
				#st.write(catcols)
				
				for n in range(len(catcols)):
					ds=data[[catcols[n],autre]].copy()
					ds=ds[ds[catcols[n]]==1]
					ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
					ds.columns=[cat,autre]
					df=df.append(ds)
							
			else: 
				df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]].copy()
			
			
			if quest.iloc[i]['graphtype']=='violin':
				if quest.iloc[i]['max']==quest.iloc[i]['max']:
					maximum=quest.iloc[i]['max']
				else:
					maximum=df[quest.iloc[i]['variable_y']].max()*1.1
									
				abscisses = df[quest.iloc[i]['variable_x']].unique().tolist()
				fig=go.Figure()
				for abscisse in abscisses:
					
					fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == abscisse],
	                            		y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == abscisse],
	                            		name=abscisse,
	                            		box_visible=True,
                           			meanline_visible=True,points="all", line_color='black', fillcolor='lightseagreen'))
				fig.update_layout(showlegend=False)
				fig.update_yaxes(range=[-0.1, maximum],title=quest.iloc[i]['ytitle'])
				st.subheader(quest.iloc[i]['title'])
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])	
			
			elif quest.iloc[i]['graphtype']=='violin2':
				#st.write(quest.iloc[i]['variable_y2'])
				df2=data[[quest.iloc[i]['variable_x2'],quest.iloc[i]['variable_y2']]].copy()
				df['Sex']=np.full(len(df),'Boys')
				df2['Sex']=np.full(len(df2),'Girls')
			
				
				fig = go.Figure()
				
				fig.add_trace(go.Box(
    				y=df[quest.iloc[i]['variable_y']],
    				x=df[quest.iloc[i]['variable_x']],
    				name='Boys',
    				marker_color='blue'
				))
				fig.add_trace(go.Box(
   				y=df2[quest.iloc[i]['variable_y2']],
    				x=df2[quest.iloc[i]['variable_x2']],
    				name='Girls',
    				marker_color='pink'
				))
				fig.update_layout(
   				 yaxis_title=quest.iloc[i]['ytitle'],
    				boxmode='group' # group together boxes of the different traces for each value of x
				)
				st.subheader(quest.iloc[i]['title'])
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])
			
			elif quest.iloc[i]['graphtype']=='bar':
				#st.write(df)
				st.subheader(quest.iloc[i]['title'])
				
				col1,col2=st.columns([1,1])

				fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
				fig1.update_layout(title_text=quest.iloc[i]['legendtitle'],font=dict(size=12),showlegend=False)	
				col1.plotly_chart(fig1,use_container_width=True)
						
				fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,legendtitle='',xaxis=quest.iloc[i]['xtitle'])
				fig1.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
				col2.plotly_chart(fig2,use_container_width=True)
				st.write(quest.iloc[i]['description'])
				#st.write(df)		
			
			elif quest.iloc[i]['graphtype']=='treemap':
				
				df['persons']=np.ones(len(df))
				st.subheader(quest.iloc[i]['title'])
				fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
				#fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])
				#st.write(df)	
			
			st.markdown("""---""")	
		
	
	elif topic=='Mapping apps':
		
		st.title('Application to visualize where the people answered what to specific questions')
		
		col1,col2=st.columns([1,1])
		
		radius=5*exp(col1.slider('Modify the size of the hexagons',5.0,10.0))
		elevation_scale=col2.slider('Modify the heights of the hexagons',10,200)
		
		reference=data[['longitude','latitude']]
		
		#st.write(reference)
		
		#st.write(radius,elevation_scale,positions.shape)		
		
				# Define a layer to display on a map
		layer = pdk.Layer(
    		"HexagonLayer",
    		reference,
    		get_position=['longitude','latitude'],
    		auto_highlight=True,
    		elevation_scale=elevation_scale,
    		pickable=True,
    		elevation_range=[0, 2000],
    		extruded=True,
    		coverage=1,
    		radius=radius,
    		tooltip=True
		)

		# Set the viewport location
		view_state = pdk.ViewState(
    		longitude=44, latitude=6.56, zoom=5, min_zoom=5, max_zoom=15, pitch=40.5, bearing=7.36,
		)
		
				
		st.subheader('Positions of all the people interviewed')
		st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
				initial_view_state=view_state,layers=[layer],tooltip=True))
		
		continues=pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
		quests=pd.read_csv('questions_map.csv',sep='\t')
		
		
		#st.write(quests)
		#st.write(len(cat_cols))
		
		L=[i for i in correlations if ('latitude'in correlations[i].tolist() or 'longitude' in correlations[i].tolist()) and (i in cat_cols or i in continues or len(data[i].unique())<20)]
		
		st.markdown("""---""")	
		
		select=st.selectbox('Select a question',[quests[i][0] for i in L])
		selection=[i for i in quests if quests[i][0]==select][0]
		st.subheader(select)
		if selection in continues:
			
			st.markdown("""---""")	
			
			threshold=st.slider('Select a threshold you want to visualize',data[selection].min(),data[selection].max())
			
			col1,col2=st.columns([1,1])
			
			positions=data[['longitude','latitude']+[selection]]
			
			layer_lower = pdk.Layer(
    		"HexagonLayer",
    		positions[positions[selection]<=threshold],
    		get_position=['longitude','latitude'],
    		auto_highlight=True,
    		elevation_scale=elevation_scale,
    		pickable=True,
    		elevation_range=[0, 2000],
    		extruded=True,
    		coverage=1,
    		radius=radius,
    		tooltip=True
		)
			
			layer_higher = pdk.Layer(
    		"HexagonLayer",
    		positions[positions[selection]>threshold],
    		get_position=['longitude','latitude'],
    		auto_highlight=True,
    		elevation_scale=elevation_scale,
    		pickable=True,
    		elevation_range=[0, 2000],
    		extruded=True,
    		coverage=1,
    		radius=radius,
    		tooltip=True
		)

			col1.subheader('Household interviewed with a lower value than '+str(threshold))
			col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
				initial_view_state=view_state,layers=[layer_lower],tooltip=True))
			col2.subheader('Household interviewed with a higher value than '+str(threshold))
			col2.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
				initial_view_state=view_state,layers=[layer_higher],tooltip=True))
		
		else:
			
			if selection in cat_cols:
				choices=[k[len(selection):] for k in data if selection in k[:len(selection)]]
			else:
				choices=data[selection].unique().tolist()
			final=st.multiselect('Select the answers you want to see on maps',choices)
			
			col1,col2=st.columns([1,1])
			
			if (selection in cat_cols) and (len(final)>0):
				#st.write('categorical')
				responses=[selection+k for k in final]
				positions=data[['longitude','latitude']+responses]
				#st.write(range(len(final)))
				for i in range(len(final)):
					layer = pdk.Layer(
		    			"HexagonLayer",
		    			positions[positions[responses[i]]==1],
			    		get_position=['longitude','latitude'],
			    		auto_highlight=True,
			    		elevation_scale=elevation_scale,
			    		pickable=True,
			    		elevation_range=[0, 2000],
			    		extruded=True,
			    		coverage=1,
			    		radius=radius,
			    		tooltip=True
					)
					if i%2==0:	
						col1.subheader('Response: '+final[i])
						col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
						initial_view_state=view_state,layers=[layer],tooltip=True))
					
					else:
						col2.subheader('Response: '+final[i])
						col2.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
						initial_view_state=view_state,layers=[layer],tooltip=True))
				
			elif len(final)>0:
				positions=data[['longitude','latitude']+[selection]]
				#st.write(positions)
				#st.write(final)
				for i in range(len(final)):
					#st.write(positions[positions[selection]==final[i]])
					layer = pdk.Layer(
		    			"HexagonLayer",
		    			positions[positions[selection]==final[i]],
			    		get_position=['longitude','latitude'],
			    		auto_highlight=True,
			    		elevation_scale=elevation_scale,
			    		pickable=True,
			    		elevation_range=[0, 2000],
			    		extruded=True,
			    		coverage=1,
			    		radius=radius,
			    		tooltip=True
					)
					#st.write(i)
					if i%2==0:	
						col1.subheader('Response: '+final[i])
						col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
						initial_view_state=view_state,layers=[layer],tooltip=True))
					
					else:
						col2.subheader('Response: '+final[i])
						col2.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',\
						initial_view_state=view_state,layers=[layer],tooltip=True))
			
			
			
				
				
					
			
	elif topic=='Display Sankey diagrams':
	
		title1.title('Sankey Diagrams')
		st.title('')
		sanks=[i for i in data.columns if data[i].dtype=='object' if 'Unnamed' not in i and len(data[i].unique())<20]
		
		#st.write(questions)
		
		sank=data[sanks].fillna('Unknown').copy()
		
		sank['ones']=np.ones(len(sank))
		
		st.markdown("""---""")
		
		st.title('Main expenses')
		
		fig=sankey_graph(sank,['regularexpence1','regularexpence2','regularexpence3'],height=1200,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.write(' - '.join(['Main expense','Second main expense','Third main expense']))
		st.plotly_chart(fig,use_container_width=True)
		
		st.markdown("""---""")
		
		
		fig=sankey_graph(sank,['challenge_1','challenge_2','challenge_3'],height=1200,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.write(' - '.join(['Main challenge','Second main challenge','Third main challenge']))
		st.plotly_chart(fig,use_container_width=True)

		st.markdown("""---""")
		
		st.write('Both for expenses and challenges we find on top Food, Water and health which are most of the time mentionned together')
		
		st.markdown("""---""")
		
		if st.checkbox('Design my own Sankey Graph'):
			
			st.markdown("""---""")
			selection=st.multiselect('Select features you want to see in the order you want them to appear',\
			 [questions[i]['question'] for i in sank if i!='ones'])
			feats=[i for i in questions if questions[i]['question'] in selection]
			
			if len(feats)>=2:
				st.write(' - '.join(selection))
				fig3=sankey_graph(sank,feats,height=1200,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3,use_container_width=True)	
		
		
		
			
	else:
		st.title('Will come later')
	
	

    
 
if __name__== '__main__':
    main()




    
