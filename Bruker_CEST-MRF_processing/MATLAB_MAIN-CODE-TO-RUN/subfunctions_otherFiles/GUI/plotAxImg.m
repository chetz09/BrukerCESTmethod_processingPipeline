% plotAxImg: Plots all axial images within the GUI given the selected
% settings.
%
%   INPUTS:
%       img             -   Struct containing images
%       roi             -   Struct containing ROI data
%       settings        -   Struct containing dynamic GUI display settings
%                           determined by user interfacing with GUI
%       si              -   Handle for UI control item in GUI that sets a
%                           label string indicating the status of the GUI
%
%   OUTPUTS:    None
%
function plotAxImg(img,roi,settings,si)

nROI=numel(roi);
[i_flds,lbls,cblims]=initPlotParams;

if strcmp(settings.plotgrp,'MRF') && settings.maskImgs
    % Mask all MRF images using dot-product loss
    mask.MRF=(img.MRF.dp>settings.dpMaskVal);
else
    mask.(settings.plotgrp)=true(img.(settings.plotgrp).size);
end
set(si,'String','Loading...')
pause(0.01) % ensures the status text above displays
if strcmp(settings.plotgrp,'ErrorMaps')
    tiledlayout(2,9);
    nexttile;axis('off')
else
    tiledlayout(2,6);
end
t=gobjects(length(i_flds.(settings.plotgrp)));
for iii = 1:length(i_flds.(settings.plotgrp))

    if strcmp(settings.plotgrp,'ErrorMaps')
        % ---- ERROR MAPS: unchanged layout ----
        t(iii)=nexttile([1 2]);
        imagesc(zeros([img.(settings.plotgrp).size,3])); hold on;
        if isfield(roi,'mask')
            if contains(i_flds.ErrorMaps{iii},'fs')
                allROImask=zeros(size(img.ErrorMaps.(i_flds.ErrorMaps{iii})));
                if isfield(roi,'nomConc')
                    for jjj=1:nROI
                        if ~isempty(roi(jjj).nomConc)
                            if ~isinf(roi(jjj).nomConc) && ~isnan(roi(jjj).nomConc)
                                allROImask=allROImask+roi(jjj).mask;
                            end
                        end
                    end
                end
            else
                allROImask=sum(reshape([roi.mask],[size(roi(1).mask),length(roi)]),3);
            end
            ei=imagesc(img.ErrorMaps.(i_flds.ErrorMaps{iii}).*mask.ErrorMaps); ...
                title(lbls.ErrorMaps.title{iii},'FontSize',18);
            if contains(i_flds.ErrorMaps{iii},'QUESP')
                try
                    set(ei,'AlphaData',allROImask.*img.other.RsqMask);
                catch
                    set(ei,'AlphaData',allROImask);
                end
            else
                set(ei,'AlphaData',allROImask);
            end
            axis('equal','off');
            cb=colorbar; clim(cblims.ErrorMaps{iii}); cb.FontSize = 14;
            cb.Label.String=lbls.ErrorMaps.cb{iii}; cb.Label.FontSize=16;
            colormap(bluewhitered);
            hold off;
        end
        if iii==4 %jump down to next plotting row
            t(iii)=nexttile; axis('off')
        end

    elseif strcmp(i_flds.(settings.plotgrp){iii},'avgZspec')
        % ---- TILE A: Z-spectrum fit ----
        t(iii)=nexttile([1 1]);
        if isfield(img.zSpec,'avgZspec')
            scatter(img.zSpec.ppm,img.zSpec.avgZspec.all.spec(settings.roiidx,:),...
                'LineWidth',1,'MarkerEdgeColor',[0.5 0.5 0.5]);
            leglbls={'Raw data'};
            hold on;

            % Plot MTR asymmetry
            if settings.showMTRasymflg
                plot(img.zSpec.MTRppm,img.zSpec.avgZspec.all.MTRasym(settings.roiidx,:),...
                    'r--*','MarkerSize',4);
                leglbls=[leglbls;{'MTR_{asym}'}];
            end

            % Plot individual pool fits + sum (exclude 'all' and 'Rex' fields)
            if settings.showFitsflg
                fitpools=fieldnames(img.zSpec.avgZspec);
                fitpools=fitpools(~strcmp(fitpools,'all') & ~strcmp(fitpools,'Rex'));
                for jjj=1:numel(fitpools)
                    plot(img.zSpec.ppm,...
                        img.zSpec.avgZspec.(fitpools{jjj}).fitSpec(settings.roiidx,:));
                end
                leglbls=[leglbls; fitpools];
                plot(img.zSpec.ppm,img.zSpec.avgZspec.all.fitSpec(settings.roiidx,:),...
                    'k-','LineWidth',2);
                leglbls=[leglbls;{'Sum'}];
            end

            legend(leglbls,'Location','southeast','FontSize',8);
            title(['Z-spectrum fit, ROI ' roi(settings.roiidx).name],'FontSize',14);
            xlabel('Offset (ppm)'); ylabel('Z(\Delta\omega)');
            xlim([min(img.zSpec.ppm) max(img.zSpec.ppm)]);
            axis('square'); set(gca,'XDir','reverse'); hold off;
        else
            axis('off');
        end

        % ---- TILE B: Rex-spectrum fit ----
        nexttile([1 1]);
        if isfield(img.zSpec,'avgZspec') && isfield(img.zSpec.avgZspec,'Rex')
            scatter(img.zSpec.RexPpm,img.zSpec.avgZspec.Rex.spec(settings.roiidx,:),...
                'LineWidth',1,'MarkerEdgeColor',[0.5 0.5 0.5]);
            leglbls_rex={'Raw data'};
            hold on;

            if settings.showFitsflg
                rexfitpools=fieldnames(img.zSpec.avgZspec.Rex);
                rexfitpools=rexfitpools(~strcmp(rexfitpools,'spec') & ...
                    ~strcmp(rexfitpools,'fitAll'));
                for jjj=1:numel(rexfitpools)
                    plot(img.zSpec.RexPpm,...
                        img.zSpec.avgZspec.Rex.(rexfitpools{jjj}).fitSpec(settings.roiidx,:));
                end
                leglbls_rex=[leglbls_rex; rexfitpools];
                plot(img.zSpec.RexPpm,img.zSpec.avgZspec.Rex.fitAll(settings.roiidx,:),...
                    'k-','LineWidth',2);
                leglbls_rex=[leglbls_rex;{'Sum'}];
            end

            legend(leglbls_rex,'Location','northeast','FontSize',8);
            title(['MTR_{Rex} fit, ROI ' roi(settings.roiidx).name],'FontSize',14);
            xlabel('Offset (ppm)');
            ylabel('MTR_{Rex} = 1/Z_{lab} - 1/Z_{ref}');
            xlim([min(img.zSpec.RexPpm) max(img.zSpec.RexPpm)]);
            axis('square'); set(gca,'XDir','reverse'); hold off;
        else
            axis('off');
        end

    elseif strcmp(i_flds.(settings.plotgrp){iii},'fitImg')
        % ---- TILE A: Z-spectrum fitted amplitude map ----
        t(iii)=nexttile([1 1]);
        imagesc(img.(settings.plotgrp).fitImg.(settings.selPool)...
            .*mask.(settings.plotgrp));
        title(['Z-fit: ' settings.selPool],'FontSize',14);
        axis('equal','off');
        cb=colorbar; clim(cblims.(settings.plotgrp){iii}); cb.FontSize=14;
        cb.Label.String=lbls.(settings.plotgrp).cb{iii}; cb.Label.FontSize=16;
        colormap(t(iii),'default');
        if isfield(roi,'coords')
            for jjj=1:length(roi)
                drawpolygon('Position',roi(jjj).coords);
            end
        end

        % ---- TILE B: Rex fitted amplitude map ----
        nexttile([1 1]);
        if isfield(img.zSpec,'RexFitImg') && ...
                isfield(img.zSpec.RexFitImg,settings.selPool)
            imagesc(img.zSpec.RexFitImg.(settings.selPool).*mask.(settings.plotgrp));
            title(['Rex-fit: ' settings.selPool],'FontSize',14);
            axis('equal','off');
            cb=colorbar; clim([0 0.3]); cb.FontSize=14;
            cb.Label.String='(s^{-1})'; cb.Label.FontSize=16;
            colormap(gca,'default');
            if isfield(roi,'coords')
                for jjj=1:length(roi)
                    drawpolygon('Position',roi(jjj).coords);
                end
            end
        else
            axis('off');
        end

    else
        % ---- ALL OTHER IMAGES: unchanged single-tile layout ----
        t(iii)=nexttile([1 2]);
        imagesc(img.(settings.plotgrp).(i_flds.(settings.plotgrp){iii})...
            .*mask.(settings.plotgrp));
        if strcmp(i_flds.(settings.plotgrp){iii},'MTRimg')
            title([lbls.(settings.plotgrp).title{iii} ', ' ...
                num2str(settings.MTRppm,'%2.1f') ' ppm'],'FontSize',18);
        else
            title(lbls.(settings.plotgrp).title{iii},'FontSize',18);
        end
        axis('equal','off');
        cb=colorbar; clim(cblims.(settings.plotgrp){iii}); cb.FontSize = 14;
        cb.Label.String=lbls.(settings.plotgrp).cb{iii}; cb.Label.FontSize=16;
        if strcmp(i_flds.(settings.plotgrp){iii},'M0img')
            colormap(t(iii),'gray');
        else
            colormap(t(iii),'default');
        end
        if isfield(roi,'coords')
            for jjj=1:length(roi)
                drawpolygon('Position',roi(jjj).coords);
            end
        end
    end

    if ~strcmp(settings.plotgrp,'ErrorMaps') && iii==3
        % Spacer tile to push bottom row to correct columns
        nexttile; axis('off')
    end
end
set(si,'String','')
end
