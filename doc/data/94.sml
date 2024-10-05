<?xml version="1.0" encoding="UTF-8"?>
<sml:SensorML xmlns:sml="http://www.opengis.net/sensorML/1.0.1" xmlns:swe="http://www.opengis.net/swe/1.0.1" xmlns:gml="http://www.opengis.net/gml" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0.1">
  <sml:member>
    <sml:System>
      <!-- ======================================= -->
      <!--               Identifiers               -->
      <!-- ======================================= -->
      <sml:identification>
        <sml:IdentifierList>
          <sml:identifier name="uniqueID">
            <sml:Term definition="urn:ogc:def:identifier:OGC:1.0:uniqueID">
              <sml:value>http://shom.fr/maregraphie/procedure/94</sml:value>
            </sml:Term>
          </sml:identifier>
          <sml:identifier name="id_shom">
            <sml:Term definition="http://">
              <sml:value>94</sml:value>
            </sml:Term>
          </sml:identifier>
          <sml:identifier name="longName">
            <sml:Term definition="urn:ogc:def:identifier:OGC:1.0:longName">
              <sml:value>BAYONNE_BOUCAU</sml:value>
            </sml:Term>
          </sml:identifier>
        </sml:IdentifierList>
      </sml:identification>
      <!-- ======================================= -->
      <!--               Classifiers               -->
      <!-- ======================================= -->
      <sml:classification>
        <sml:ClassifierList>
          <sml:classifier name="value">
            <sml:Term definition="http://shom.fr/maregraphie/id_shom">
              <sml:value>94</sml:value>
            </sml:Term>
          </sml:classifier>
          <sml:classifier name="label">
            <sml:Term definition="http://shom.fr/maregraphie/label">
              <sml:value>BAYONNE_BOUCAU</sml:value>
            </sml:Term>
          </sml:classifier>
        </sml:ClassifierList>
      </sml:classification>
      <!-- ======================================= -->
      <!--            Constraints              -->
      <!-- =======================================  -->
      <sml:validTime>
        <gml:TimePeriod gml:id="documentValidTime">
          <gml:beginPosition>2050-01-01</gml:beginPosition>
          <gml:endPosition indeterminatePosition="now"/>
        </gml:TimePeriod>
      </sml:validTime>
      <sml:legalConstraint>
        <sml:Rights>
          <sml:documentation>
            <sml:Document>
              <gml:description>Voir les conditions générales d'utilsation sur l'espace de diffusion.</gml:description>
            </sml:Document>
          </sml:documentation>
        </sml:Rights>
      </sml:legalConstraint>
      <!-- ======================================= -->
      <!--            Characteristics              -->
      <!--            in capapabilities...         -->
      <!-- =======================================  -->
      <sml:capabilities name="characterics">
        <swe:DataRecord>
          <swe:field name="ville_d_hebergement">
            <swe:Text definition="http://shom.fr/maregraphie/ville_d_hebergement">
              <swe:value>Bayonne Boucau</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="fuseau_horaire">
            <swe:Text definition="http://shom.fr/maregraphie/fuseau_horaire">
              <swe:value>0</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="longitude">
            <swe:Quantity definition="http://shom.fr/maregraphie/longitude">
              <swe:value>-1.51483</swe:value>
            </swe:Quantity>
          </swe:field>
          <swe:field name="latitude">
            <swe:Quantity definition="http://shom.fr/maregraphie/latitude">
              <swe:value>43.52732</swe:value>
            </swe:Quantity>
          </swe:field>
          <swe:field name="sect_geographique">
            <swe:Text definition="http://shom.fr/maregraphie/sect_geographique">
              <swe:value>FM</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="date_prem_obs">
            <swe:Category definition="http://shom.fr/maregraphie/date_prem_obs">
              <swe:value>1967-05-22</swe:value>
            </swe:Category>
          </swe:field>
          <swe:field name="descriptif_capteur">
            <swe:Text definition="http://shom.fr/maregraphie/descriptif_capteur">
              <swe:value>https://refmar.shom.fr/maregraphe-radar-sans-contact-krohne-optiwave-7300c</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="collocalisation">
            <swe:Text definition="http://shom.fr/maregraphie/collocalisation">
              <swe:value>https://www.sonel.org/spip.php?page=gps&amp;idStation=3192</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="etat_maregraphe">
            <swe:Text definition="http://shom.fr/maregraphie/etat_maregraphe">
              <swe:value>OK</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="info_maregraphe">
            <swe:Text definition="http://shom.fr/maregraphie/info_maregraphe">
              <swe:value>https://refmar.shom.fr/donnees/94</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="journal_de_bord">
            <swe:Text definition="http://shom.fr/maregraphie/journal_de_bord">
              <swe:value>https://refmar.shom.fr/donnees/94</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="spm">
            <swe:Text definition="http://shom.fr/maregraphie/spm">
              <swe:value>BOUCAU-BAYONNE</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="zh_ref">
            <swe:Text definition="http://shom.fr/maregraphie/zh_ref">
              <swe:value>-2.143</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="nom_ref">
            <swe:Text definition="http://shom.fr/maregraphie/nom_ref">
              <swe:value>IGN69</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="zero_hydro">
            <swe:Text definition="http://shom.fr/maregraphie/zero_hydro">
              <swe:value>zero_hydrographique</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="reseau">
            <swe:Text definition="http://shom.fr/maregraphie/reseau">
              <swe:value>RONIM</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="id_ram">
            <swe:Text definition="http://shom.fr/maregraphie/id_ram">
              <swe:value>Boucau-Bayonne</swe:value>
            </swe:Text>
          </swe:field>
          <swe:field name="gestionnaire">
            <swe:Text definition="http://shom.fr/maregraphie/gestionnaire">
              <swe:value>Shom</swe:value>
            </swe:Text>
          </swe:field>
        </swe:DataRecord>
      </sml:capabilities>
      <!-- ================================= -->
      <!--            Capabilities           -->
      <!-- ================================= -->
      <sml:capabilities name="offerings">
        <swe:SimpleDataRecord>
          <swe:field name="Offering_for_sensor">
            <swe:Text definition="urn:ogc:def:identifier:OGC:offeringID">
              <swe:value>http://shom.fr/maregraphie/offering/94</swe:value>
            </swe:Text>
          </swe:field>
        </swe:SimpleDataRecord>
      </sml:capabilities>
      <sml:capabilities name="featuresOfInterest">
        <swe:SimpleDataRecord>
          <swe:field name="featureOfInterestID">
            <swe:Text>
              <swe:value>http://shom.fr/maregraphie/featureOfInterest/94</swe:value>
            </swe:Text>
          </swe:field>
        </swe:SimpleDataRecord>
      </sml:capabilities>
      <sml:capabilities name="organisme">
        <swe:DataRecord definition="http://shom.fr/maregraphie/organisme">
          
          <swe:field name="Shom">
            <swe:DataRecord definition="http://shom.fr/maregraphie/organisme">
              <swe:field name="nom">
                <swe:Text definition="http://shom.fr/maregraphie/nom_organisme">
                  <swe:value>Shom</swe:value>
                </swe:Text>
              </swe:field>
              <swe:field name="logo">
                <swe:Text definition="http://shom.fr/maregraphie/logo">
                  <swe:value>https://services.data.shom.fr/static/logo/DDM/logo_SHOM.png</swe:value>
                </swe:Text>
              </swe:field>
              <swe:field name="URL">
                <swe:Text definition="http://shom.fr/maregraphie/lien">
                  <swe:value>https://www.shom.fr/</swe:value>
                </swe:Text>
              </swe:field>
            </swe:DataRecord>
          </swe:field>
          <swe:field name="Région Nouvelle-Aquitaine">
            <swe:DataRecord definition="http://shom.fr/maregraphie/organisme">
              <swe:field name="nom">
                <swe:Text definition="http://shom.fr/maregraphie/nom_organisme">
                  <swe:value>Région Nouvelle-Aquitaine</swe:value>
                </swe:Text>
              </swe:field>
              <swe:field name="logo">
                <swe:Text definition="http://shom.fr/maregraphie/logo">
                  <swe:value>https://services.data.shom.fr/static/logo/DDM/logo_CR_Aquitaine.png</swe:value>
                </swe:Text>
              </swe:field>
              <swe:field name="URL">
                <swe:Text definition="http://shom.fr/maregraphie/lien">
                  <swe:value>https://www.nouvelle-aquitaine.fr/</swe:value>
                </swe:Text>
              </swe:field>
            </swe:DataRecord>
          </swe:field>
          
        </swe:DataRecord>
      </sml:capabilities>
      <!-- ============================ -->
      <!--           Contacts           -->
      <!-- ============================ -->
      
      <sml:contact>
        <sml:ContactList>
          <sml:member>
            <sml:ResponsibleParty>
              <sml:individualName>SHOM</sml:individualName>
              <sml:organizationName>SHOM</sml:organizationName>
              <sml:contactInfo>
                <sml:phone>
                  <sml:voice>02 56 31 24 26</sml:voice>
                </sml:phone>
                <sml:address>
                  <sml:deliveryPoint>13 rue du chatellier</sml:deliveryPoint>
                  <sml:city>BREST</sml:city>
                  <sml:postalCode>29200</sml:postalCode>
                  <sml:country>France</sml:country>
                  <sml:electronicMailAddress>refmar@shom.fr</sml:electronicMailAddress>
                </sml:address>
                <sml:onlineResource xlink:href="http://shom.fr/maregraphie"/>
              </sml:contactInfo>
            </sml:ResponsibleParty>
          </sml:member>
        </sml:ContactList>
      </sml:contact>
      
      <!-- ============================ -->
      <!--         Documentation        -->
      <!-- ============================ -->
      <!-- ============================ -->
      <!--            Position          -->
      <!-- ============================ -->
      <sml:position name="sensorPosition">
        <swe:Position fixed="true" referenceFrame="urn:ogc:def:crs:EPSG::4326">
          <swe:location>
            <swe:Vector gml:id="STATION_LOCATION">
              <swe:coordinate name="latitude">
                <swe:Quantity axisID="x">
                  <swe:uom code="degree"/>
                  <swe:value>43.52732</swe:value>
                </swe:Quantity>
              </swe:coordinate>
              <swe:coordinate name="longitude">
                <swe:Quantity axisID="y">
                  <swe:uom code="degree"/>
                  <swe:value>-1.51483</swe:value>
                </swe:Quantity>
              </swe:coordinate>
            </swe:Vector>
          </swe:location>
        </swe:Position>
      </sml:position>
      <!-- =============================== -->
      <!--              Inputs             -->
      <!-- =============================== -->
      <sml:inputs>
        <sml:InputList>
          <sml:input name="observedProperty_WaterHeight">
            <swe:ObservableProperty definition="http://shom.fr/maregraphie/observedProperty/WaterHeight"/>
          </sml:input>
        </sml:InputList>
      </sml:inputs>
      <!-- =============================== -->
      <!--              Outputs            -->
      <!-- =============================== -->
      <sml:outputs>
        <sml:OutputList>
          <sml:output name="observedProperty_WaterHeight_1">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/1"/>
          </sml:output>
          <sml:output name="observedProperty_WaterHeight_2">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/2"/>
          </sml:output>
          <sml:output name="observedProperty_WaterHeight_3">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/3"/>
          </sml:output>
          <sml:output name="observedProperty_WaterHeight_4">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/4"/>
          </sml:output>
          <sml:output name="observedProperty_WaterHeight_5">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/5"/>
          </sml:output>
          <sml:output name="observedProperty_WaterHeight_6">
            <swe:Count definition="http://shom.fr/maregraphie/observedProperty/WaterHeight/6"/>
          </sml:output>
        </sml:OutputList>
      </sml:outputs>
      <!-- =============================== -->
      <!--              History            -->
      <!-- =============================== -->
      <sml:history xlink:title="observatory_logbook_events">
        <sml:EventList>
          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:58:00.000Z</sml:date>
              <gml:description>
	Changement de la valeur du paramètre D: 659 au lieu de 660.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:00:01.000Z</sml:date>
              <gml:description>
	L&apos;interrogation automatique de ce jour a fonctionné correctement. La ligne est de nouveau disponible.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:00:03.000Z</sml:date>
              <gml:description>
	La récupération automatique du week-end ne s&apos;est pas faite.
	Avons contacté les partenaires. Pb avec la ligne téléphonique à suivre ...
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:58:04.000Z</sml:date>
              <gml:description>
	Relance du MCN. Archivage des données à 10 min et HH.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:58:02.000Z</sml:date>
              <gml:description>
	Modernisation du marégraphe - Remplacement de l&apos;ancien marégraphe par un nouveau capteur et une nouvelle centrale ELTA.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2015-12-07">
            <sml:Event>
              <sml:date>2015-12-07T14:03:01.000Z</sml:date>
              <gml:description>
	Changment de la côte du D suite au dernier nivellement.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:57:04.000Z</sml:date>
              <gml:description>
	Installation d&apos;un marégraphe côtier numérique (MCN): capteur ultrason et centrale MORS HT200.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2014-10-10">
            <sml:Event>
              <sml:date>2014-10-10T06:22:02.000Z</sml:date>
              <gml:description>
	Une équipe du SHOM a installé une transmission satellite Meteosat sur le marégraphe RONIM de Bayonne. Cet équipement améliore la robustesse des transmissions temps réel.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:02:04.000Z</sml:date>
              <gml:description>
	Les données validées jusqu&apos;au 30 mai 2011 sont disponibles sur le serveur FTP
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2012-02-27">
            <sml:Event>
              <sml:date>2012-02-27T14:59:01.000Z</sml:date>
              <gml:description>
	Au tracé, le bas des courbes est un peu déformé. Un mail a été envoyé aux partenaires demandant un prochain désenvasement.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:59:05.000Z</sml:date>
              <gml:description>
	Contact avec les partenaires au sujet du pb de ligne.
	Coupure de la ligne. La région tente de récupérer l&apos;abonnement au même numéro, à suivre ...
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2014-04-25">
            <sml:Event>
              <sml:date>2014-04-25T13:58:05.000Z</sml:date>
              <gml:description>
	Installation nouvelle centrale
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:59:02.000Z</sml:date>
              <gml:description>
	Changement de la centrale
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:59:04.000Z</sml:date>
              <gml:description>
	Relance du MCN. Archivage des données.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2014-04-25">
            <sml:Event>
              <sml:date>2014-04-25T13:59:02.000Z</sml:date>
              <gml:description>
	Changement de capteur.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2014-04-25">
            <sml:Event>
              <sml:date>2014-04-25T14:00:01.000Z</sml:date>
              <gml:description>
	Etalonnage du capteur + installation interrupteur GSM
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2014-04-25">
            <sml:Event>
              <sml:date>2014-04-25T14:00:03.000Z</sml:date>
              <gml:description>
	Nettoyage du puits.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:01:03.000Z</sml:date>
              <gml:description>
	Le signal mesuré lors des basses mers pourrait traduire un envasement du puits de tranquillisation ou autre problème d&apos;acquisition. La mesure est considérée douteuse en attente d&apos;investigations.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-09-15">
            <sml:Event>
              <sml:date>2011-09-15T08:34:01.000Z</sml:date>
              <gml:description>
	Nettoyage du puits effectué ce jour.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T06:59:00.000Z</sml:date>
              <gml:description>
	Problème avec la centrale et le modem (certainement dû à un orage).
	Impossibilité pour le moment d&apos;interroger le MCN et de récupérer les données.
	Une nouvelle centrale vient d&apos;être envoyée. Tout devrait rentrer dans l&apos;ordre prochainement.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2012-11-08">
            <sml:Event>
              <sml:date>2012-11-08T16:02:00.000Z</sml:date>
              <gml:description>
	Suite à un problème technique, les données brutes fournies par le marégraphe de Bayonne_Boucau ne sont pas disponibles entre 03h00 et 15h30 (TU).

	Néanmoins, les données ont été enregistrées. Dès que possible, elles seront mises en ligne comme données validées (à pas de temps de 10 minutes et horaire). 
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:00:05.000Z</sml:date>
              <gml:description>
	Relance du MCN. Archivage des données.
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2015-07-02">
            <sml:Event>
              <sml:date>2015-07-02T08:11:05.000Z</sml:date>
              <gml:description>
	Controle à la sonde lumineuse effectué ce jour.

	Moyenne des écarts à BM : -0.15 cm (écart type : 0.45 cm)

	Moyenne des écarts à PM : -0.26 cm (écart type : 0.27 cm)
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2011-07-06">
            <sml:Event>
              <sml:date>2011-07-06T07:01:01.000Z</sml:date>
              <gml:description>
	Les données validées jusqu&apos;au 10/01/2011 sont disponibles
</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2016-11-18">
            <sml:Event>
              <sml:date>2016-11-18T09:18:57.358Z</sml:date>
              <gml:description>Problème dans le puits. L&apos;eau stagne à marée basse. Le signal de marée est tronqué à basse mer. </gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2017-01-24">
            <sml:Event>
              <sml:date>2017-01-24T15:22:23.707Z</sml:date>
              <gml:description>La liaison RTC avec le MCN est en avarie depuis le 3 janvier 2017. L&apos;analyse est en cours.</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2017-01-27">
            <sml:Event>
              <sml:date>2017-01-27T15:18:07.028Z</sml:date>
              <gml:description>Retour à la normale, bon fonctionnement général à l&apos;exception des basses mers qui sont parfois tronquées (depend du coefficient) du fait d&apos;un problème de vidage du puits</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2017-04-03">
            <sml:Event>
              <sml:date>2017-04-03T09:48:29.062Z</sml:date>
              <gml:description>Marégraphe en panne, investigation en cours</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2017-04-21">
            <sml:Event>
              <sml:date>2017-04-21T14:07:22.475Z</sml:date>
              <gml:description>Le MCN a retrouvé son alimentation suite à une permutation de batterie.</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2018-11-26">
            <sml:Event>
              <sml:date>2018-11-26T18:35:58.000Z</sml:date>
              <gml:description>Erreur de transmission de données temps réel : données accessibles en différé entre le 26/11 et le 27/11</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2019-02-21">
            <sml:Event>
              <sml:date>2019-02-21T10:44:59.000Z</sml:date>
              <gml:description>Erreur serveur Shom entre le 20/02/19 et le 21/02/19 : les données 1 minute n&apos;ont pas été reçues </gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2019-06-27">
            <sml:Event>
              <sml:date>2019-06-27T09:56:29.000Z</sml:date>
              <gml:description>Données temps réelles manquantes entre 26/06/2019 et 27/06/2019 suite à une panne réseau</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2020-06-02">
            <sml:Event>
              <sml:date>2020-06-02T10:55:39.000Z</sml:date>
              <gml:description>panne secteur au marégraphe, une intervention de notre partenaire a été programmée.</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2020-06-08">
            <sml:Event>
              <sml:date>2020-06-08T16:32:24.000Z</sml:date>
              <gml:description>bon fonctionnement</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2020-01-08">
            <sml:Event>
              <sml:date>2020-01-08T15:32:04.000Z</sml:date>
              <gml:description>Intervention : changement d&apos;antenne pour la transmission de données par satellite. MCN complètement opérationnel</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2021-11-25">
            <sml:Event>
              <sml:date>2021-11-25T08:36:42.000Z</sml:date>
              <gml:description>Afin de répondre aux besoins des utilisateurs, le système de transmission des données du réseau RONIM évolue. Cette évolution permet d’améliorer la mise à disposition/visualisation des données brutes HF tout en réduisant les lacunes d&apos;observation.</gml:description>
            </sml:Event>
          </sml:member>          <sml:member name="logbook-2021-11-25">
            <sml:Event>
              <sml:date>2021-11-25T08:37:08.000Z</sml:date>
              <gml:description>Travaux de maintenance les 24 et 25/11/2021.</gml:description>
            </sml:Event>
          </sml:member>
        </sml:EventList>
      </sml:history>
    </sml:System>
  </sml:member>
</sml:SensorML>